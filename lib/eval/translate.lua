require 'nn'
require 'string'
require 'nngraph'

require 'lib.utils.dict'

local Beam = require 'lib.eval.beam'

local Encoder = require 'lib.encoder'
local BiEncoder = require 'lib.biencoder'
local Decoder = require 'lib.decoder'
local Generator = require 'lib.generator'
local Data = require 'lib.data'

local constants = require 'lib.utils.constants'
local cuda = require 'lib.utils.cuda'
local path = require 'pl.path'

local checkpoint = nil
local models = {}
local opt = {}

local src_dict
local targ_dict

local phrase_table = {}


local function absolute_path(file_path, resources_dir)
  local function isempty(s)
    return s == nil or s == ''
  end

  if not isempty(resources_dir) and not isempty(file_path) then
    file_path = path.join(resources_dir, file_path)
  end

  if not isempty(file_path) then
    assert(path.exists(file_path), 'Path does not exist: ' .. file_path)
  end

  return file_path
end

local function load_phrase_table(file_path)
  local f = io.open(file_path, 'r')

  if f == nil then
    error('Failed to open file ' .. file_path)
  end

  local function strip(s)
    return s:gsub("^%s+",""):gsub("%s+$","")
  end

  local pt = {}

  for line in f:lines() do
    local c = line:split("|||")
    pt[strip(c[1])] = c[2]
  end

  f:close()

  return pt
end

local function init(args, resources_dir)
  opt = args
  cuda.init(opt)

  opt.model = absolute_path(opt.model, resources_dir)
  print('Loading ' .. opt.model .. '...')
  checkpoint = torch.load(opt.model)

  if checkpoint.options.brnn then
    models.encoder = BiEncoder.new({
      max_batch_size = opt.batch,
      max_sent_length = opt.max_sent_l,
      num_layers = checkpoint.options.num_layers,
      rnn_size = checkpoint.options.rnn_size,
      mask_padding = true
    }, checkpoint.options.brnn_merge, checkpoint.nets.encoder, checkpoint.nets.encoder_bwd)
  else
    models.encoder = Encoder.new({
      max_batch_size = opt.batch,
      max_sent_length = opt.max_sent_l,
      num_layers = checkpoint.options.num_layers,
      rnn_size = checkpoint.options.rnn_size,
      mask_padding = true
    }, checkpoint.nets.encoder)
  end

  models.decoder = Decoder.new({
    max_batch_size = opt.batch,
    rnn_size = checkpoint.options.rnn_size,
    num_layers = checkpoint.options.num_layers,
    input_feed = checkpoint.options.input_feed,
    mask_padding = true
  }, checkpoint.nets.decoder)

  models.generator = Generator.new({}, checkpoint.nets.generator)

  models.encoder:evaluate()
  models.decoder:evaluate()
  models.generator:evaluate()

  cuda.convert(models.encoder)
  cuda.convert(models.decoder)
  cuda.convert(models.generator)

  src_dict = checkpoint.dicts.src
  targ_dict = checkpoint.dicts.targ

  if opt.srctarg_dict:len() > 0 then
    opt.srctarg_dict = absolute_path(opt.srctarg_dict, resources_dir)
    phrase_table = load_phrase_table(opt.srctarg_dict)
  end
end

local function build_data(src_batch, gold_batch)
  local src = {}
  local targ

  if gold_batch ~= nil then
    targ = {}
  end

  for b = 1, #src_batch do
    table.insert(src, src_dict:convert_to_idx(src_batch[b], false))

    if targ ~= nil then
      table.insert(targ, targ_dict:convert_to_idx(gold_batch[b], true))
    end
  end

  return Data.new(src, targ)
end

local function build_target_tokens(pred, src, attn)
  local tokens = targ_dict:convert_to_labels(pred, constants.EOS)

  if opt.replace_unk then
    for i = 1, #tokens do
      if tokens[i] == constants.UNK_WORD then
        local _, max_index = attn[i]:max(1)
        local source = src[max_index[1]]

        if phrase_table[source] ~= nil then
          tokens[i] = phrase_table[source]
        else
          tokens[i] = source
        end
      end
    end
  end

  return tokens
end

local function translate_batch(batch)
  -- resize preallocated tensors to handle new batch size
  models.encoder:resize_proto(batch.size)
  models.decoder:resize_proto(opt.beam * batch.size)

  -- also forget previous padding module on the decoder
  models.decoder:reset()

  local enc_states, context = models.encoder:forward(batch)

  local gold_score
  if batch.target_input ~= nil then
    if batch.size > 1 then
      models.decoder:reset(batch.source_size, batch.source_length)
    end
    gold_score = models.decoder:compute_score(batch, enc_states, context, models.generator)
  end

  -- expand tensors for each beam
  context = context
    :contiguous()
    :view(1, batch.size, batch.source_length, checkpoint.options.rnn_size)
    :expand(opt.beam, batch.size, batch.source_length, checkpoint.options.rnn_size)
    :contiguous()
    :view(opt.beam * batch.size, batch.source_length, checkpoint.options.rnn_size)

  for j = 1, #enc_states do
    enc_states[j] = enc_states[j]
      :view(1, batch.size, checkpoint.options.rnn_size)
      :expand(opt.beam, batch.size, checkpoint.options.rnn_size)
      :contiguous()
      :view(opt.beam * batch.size, checkpoint.options.rnn_size)
  end

  local remaining_sents = batch.size
  local batch_idx = {}

  local beam = {}

  for b = 1, batch.size do
    table.insert(beam, Beam.new(opt.beam, opt.max_sent_l))
    table.insert(batch_idx, b)
  end

  local i = 1

  local dec_out
  local dec_states = enc_states

  while remaining_sents > 0 and i < opt.max_sent_l do
    i = i + 1

    -- prepare decoder input
    local input = torch.IntTensor(opt.beam, remaining_sents)
    local source_sizes = torch.IntTensor(remaining_sents)

    for b = 1, batch.size do
      if not beam[b].done then
        local idx = batch_idx[b]
        source_sizes[idx] = batch.source_size[b]
        input[{{}, idx}]:copy(beam[b]:get_current_state())
      end
    end

    input = input:view(opt.beam * remaining_sents)

    if batch.size > 1 then
      models.decoder:reset(source_sizes, batch.source_length, opt.beam)
    end

    dec_out, dec_states = models.decoder:forward_one(input, dec_states, context, dec_out)

    local out = models.generator:forward_one(dec_out)

    out = out:view(opt.beam, remaining_sents, out:size(2)):transpose(1, 2):contiguous()

    local softmax_out = models.decoder.softmax_attn.output:view(opt.beam, remaining_sents, -1)
    local new_remaining_sents = remaining_sents

    for b = 1, batch.size do
      if not beam[b].done then
        local idx = batch_idx[b]

        if beam[b]:advance(out[idx], softmax_out[{{}, idx}]) then
          new_remaining_sents = new_remaining_sents - 1
          batch_idx[b] = 0
        end

        for j = 1, #dec_states do
          local view = dec_states[j]
            :view(opt.beam, remaining_sents, checkpoint.options.rnn_size)
          view[{{}, idx}] = view[{{}, idx}]:index(1, beam[b]:get_current_origin())
        end
      end
    end

    if new_remaining_sents > 0 and new_remaining_sents ~= remaining_sents then
      -- update sentence indices within the batch and mark sentences to keep
      local to_keep = {}
      local new_idx = 1
      for b = 1, #batch_idx do
        local idx = batch_idx[b]
        if idx > 0 then
          table.insert(to_keep, idx)
          batch_idx[b] = new_idx
          new_idx = new_idx + 1
        end
      end

      to_keep = torch.LongTensor(to_keep)

      -- update rnn_state and context
      for j = 1, #dec_states do
        dec_states[j] = dec_states[j]
          :view(opt.beam, remaining_sents, checkpoint.options.rnn_size)
          :index(2, to_keep)
          :view(opt.beam*new_remaining_sents, checkpoint.options.rnn_size)
      end

      dec_out = dec_out
        :view(opt.beam, remaining_sents, checkpoint.options.rnn_size)
        :index(2, to_keep)
        :view(opt.beam*new_remaining_sents, checkpoint.options.rnn_size)

      context = context
        :view(opt.beam, remaining_sents, batch.source_length, checkpoint.options.rnn_size)
        :index(2, to_keep)
        :view(opt.beam*new_remaining_sents, batch.source_length, checkpoint.options.rnn_size)
    end

    remaining_sents = new_remaining_sents
    collectgarbage()
  end

  local all_hyp = {}
  local all_attn = {}
  local all_scores = {}

  for b = 1, batch.size do
    local scores, ks = beam[b]:sort_best()

    local hyp_batch = {}
    local attn_batch = {}
    local scores_batch = {}

    for n = 1, opt.n_best do
      local hyp, attn = beam[b]:get_hyp(ks[n])

      -- remove unnecessary values from the attention vectors
      for j = 1, #attn do
        local size = batch.source_size[b]
        attn[j] = attn[j]:narrow(1, batch.source_length - size + 1, size)
      end

      table.insert(hyp_batch, hyp)
      table.insert(attn_batch, attn)
      table.insert(scores_batch, scores[n])
    end

    table.insert(all_hyp, hyp_batch)
    table.insert(all_attn, attn_batch)
    table.insert(all_scores, scores_batch)
  end

  return all_hyp, all_scores, all_attn, gold_score
end

local function translate(src_batch, gold_batch)
  local data = build_data(src_batch, gold_batch)
  local batch = data:get_batch()

  local pred, pred_score, attn, gold_score = translate_batch(batch)

  local pred_batch = {}
  local info_batch = {}

  for b = 1, batch.size do
    table.insert(pred_batch, build_target_tokens(pred[b][1], src_batch[b], attn[b][1]))

    local info = {}
    info.score = pred_score[b][1]
    info.n_best = {}

    if gold_score ~= nil then
      info.gold_score = gold_score[b]
    end

    if opt.n_best > 1 then
      for n = 1, opt.n_best do
        info.n_best[n] = {}
        info.n_best[n].tokens = build_target_tokens(pred[b][n], src_batch[b], attn[b][n])
        info.n_best[n].score = pred_score[b][n]
      end
    end

    table.insert(info_batch, info)
  end

  return pred_batch, info_batch
end

return {
  init = init,
  translate = translate
}
