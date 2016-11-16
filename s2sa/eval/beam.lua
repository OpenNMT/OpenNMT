require 'nn'
require 'string'
require 'nngraph'

require 's2sa.utils.dict'

local Encoder = require 's2sa.encoder'
local BiEncoder = require 's2sa.biencoder'
local Decoder = require 's2sa.decoder'
local Generator = require 's2sa.generator'

local table_utils = require 's2sa.utils.table_utils'
local constants = require 's2sa.utils.constants'
local cuda = require 's2sa.utils.cuda'
local State = require 's2sa.eval.state'
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


local function get_max_length(tokens)
  local max = 0
  local sizes = {}
  for i = 1, #tokens do
    local len = #tokens[i]
    max = math.max(max, len)
    table.insert(sizes, len)
  end
  return max, sizes
end

local function build_data(src_batch, gold_batch)
  local batch = {}

  batch.size = #src_batch
  batch.source_length, batch.source_size = get_max_length(src_batch)

  assert(batch.source_length <= opt.max_sent_l, 'maximum sentence length reached')

  batch.source_input = torch.IntTensor(batch.source_length, batch.size):fill(constants.PAD)

  if gold_batch ~= nil then
    batch.target_length = get_max_length(gold_batch) + 1 -- for <s> or </s>
    batch.target_input = torch.IntTensor(batch.target_length, batch.size):fill(constants.PAD)
    batch.target_output = torch.IntTensor(batch.target_length, batch.size):fill(constants.PAD)
  end

  for b = 1, batch.size do
    local source_input = src_dict:convert_to_idx(src_batch[b], false)
    local source_offset = batch.source_length - batch.source_size[b] + 1

    -- pad on the left
    batch.source_input[{{source_offset, batch.source_length}, b}]:copy(source_input)

    if gold_batch ~= nil then
      local target = targ_dict:convert_to_idx(gold_batch[b], true)
      local target_input = target:narrow(1, 1, batch.target_length)
      local target_output = target:narrow(1, 2, batch.target_length)

      batch.target_input[{{1, batch.target_length}, b}]:copy(target_input)
      batch.target_output[{{1, batch.target_length}, b}]:copy(target_output)
    end
  end

  batch.source_input = cuda.convert(batch.source_input)

  if gold_batch ~= nil then
    batch.target_input = cuda.convert(batch.target_input)
    batch.target_output = cuda.convert(batch.target_output)
  end

  return batch
end

local function build_target_tokens(pred, src, attn)
  local tokens = targ_dict:convert_to_labels(pred)

  -- ignore <s> and </s>
  table.remove(tokens)
  table.remove(tokens, 1)

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

local function generate_beam(batch)
  -- resize preallocated preallocated tensors to handle new batch size
  models.encoder:resize_proto(batch.size)
  models.decoder:resize_proto(opt.beam * batch.size)

  -- also forget previous padding module on the decoder
  models.decoder:reset()

  local enc_states, context = models.encoder:forward(batch)

  local gold_score
  if batch.target_input ~= nil then
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

  -- setup beam search
  local initial_state = State.initial(constants.BOS)

  local prev_ks = torch.LongTensor(batch.size, opt.max_sent_l, opt.beam):fill(constants.PAD)
  local next_ys = torch.LongTensor(batch.size, opt.max_sent_l, opt.beam):fill(constants.PAD)
  local scores = torch.FloatTensor(batch.size, opt.max_sent_l, opt.beam):zero()

  local attn_weights = {}
  local states = {}
  for b = 1, batch.size do
    table.insert(attn_weights, {{initial_state}}) -- store attn weights
    table.insert(states, {{initial_state}}) -- store predicted word idx
    next_ys[b][1][1] = State.next(initial_state)
  end

  local out_float = torch.FloatTensor()

  local done = {}
  local found_eos = {}

  local end_score = {}
  local end_hyp = {}
  local end_attn_weights = {}

  local max_score = {}
  local max_hyp = {}
  local max_k = {}
  local max_attn_weights = {}

  local remaining_sents = batch.size
  local batch_idx = {}

  for b = 1, batch.size do
    done[b] = false
    found_eos[b] = false
    max_score[b] = -1e9
    end_score[b] = -1e9

    table.insert(max_k, 1)
    table.insert(batch_idx, b)
  end

  local i = 1

  local dec_out
  local dec_states = enc_states

  while remaining_sents > 0 and i < opt.max_sent_l do
    i = i + 1

    -- prepare decoder input
    local input = torch.IntTensor(opt.beam, remaining_sents)
    local source_sizes = {}

    for b = 1, batch.size do
      if not done[b] then
        local idx = batch_idx[b]
        table.insert(source_sizes, batch.source_size[b])

        states[b][i] = {}
        attn_weights[b][i] = {}

        local y = next_ys[b]:narrow(1, i - 1, 1):squeeze()
        if opt.beam == 1 then
          input[{{}, idx}]:copy(torch.IntTensor({y}))
        else
          input[{{}, idx}]:copy(y)
        end
      end
    end

    input = input:view(opt.beam * remaining_sents)

    if batch.size > 1 then
      models.decoder:reset(source_sizes, batch.source_length, opt.beam)
    end

    dec_out, dec_states = models.decoder:forward_one(input, dec_states, context, dec_out)

    local out = models.generator:forward_one(dec_out)

    out = out:view(opt.beam, remaining_sents, out:size(2)):transpose(1, 2)
    out_float:resize(out:size()):copy(out)

    local softmax_out = models.decoder.softmax_attn.output:view(opt.beam, remaining_sents, -1)
    local new_remaining_sents = remaining_sents

    for b = 1, batch.size do
      if done[b] == false then
        local idx = batch_idx[b]
        for k = 1, opt.beam do
          State.disallow(out_float[idx]:select(1, k))
          out_float[idx][k]:add(scores[b][i-1][k])
        end

        -- All the scores available.

        local flat_out = out_float[idx]:view(-1)
        if i == 2 then
          flat_out = out_float[idx][1] -- all outputs same for first batch
        end

        for k = 1, opt.beam do
          while true do
            local score, index = flat_out:max(1)
            score = score[1]
            local prev_k, y_i = table_utils.flat_to_rc(out_float[idx], index[1])
            states[b][i][k] = State.advance(states[b][i-1][prev_k], y_i)
            local diff = true
            for k2 = 1, k-1 do
              if State.same(states[b][i][k2], states[b][i][k]) then
                diff = false
              end
            end

            if i < 2 or diff then
              attn_weights[b][i][k] = State.advance(attn_weights[b][i-1][prev_k], softmax_out[prev_k][idx]:clone())
              prev_ks[b][i][k] = prev_k
              next_ys[b][i][k] = y_i
              scores[b][i][k] = score
              flat_out[index[1]] = -1e9
              break -- move on to next k
            end
            flat_out[index[1]] = -1e9
          end
        end

        for j = 1, #dec_states do
          local view = dec_states[j]
            :view(opt.beam, remaining_sents, checkpoint.options.rnn_size)
          view[{{}, idx}] = view[{{}, idx}]:index(1, prev_ks[b][i])
        end

        end_hyp[b] = states[b][i][1]
        end_score[b] = scores[b][i][1]
        end_attn_weights[b] = attn_weights[b][i][1]
        if end_hyp[b][#end_hyp[b]] == constants.EOS then
          done[b] = true
          found_eos[b] = true
          new_remaining_sents = new_remaining_sents - 1
          batch_idx[b] = 0
        else
          for k = 1, opt.beam do
            local possible_hyp = states[b][i][k]
            if possible_hyp[#possible_hyp] == constants.EOS then
              found_eos[b] = true
              if scores[b][i][k] > max_score[b] then
                max_hyp[b] = possible_hyp
                max_score[b] = scores[b][i][k]
                max_k[b] = k
                max_attn_weights[b] = attn_weights[b][i][k]
              end
            end
          end
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

  local states_res = {}
  local scores_res = {}

  for b = 1, batch.size do
    if opt.simple == 1 or end_score[b] > max_score[b] or not found_eos[b] then
      max_hyp[b] = end_hyp[b]
      max_score[b] = end_score[b]
      max_attn_weights[b] = end_attn_weights[b]
      max_k[b] = 1
    end

    -- remove unnecessary values from the attention vectors
    for j = 2, #max_attn_weights[b] do
      local size = batch.source_size[b]
      max_attn_weights[b][j] = max_attn_weights[b][j]:narrow(1, batch.source_length-size+1, size)
    end

    local sent_len = #max_hyp[b]

    table.insert(states_res, states[b][sent_len])
    table.insert(scores_res, scores[b][sent_len])
  end

  return max_hyp, max_score, max_attn_weights, states_res, scores_res, gold_score
end

local function search(src_batch, gold_batch)
  local batch = build_data(src_batch, gold_batch)

  local pred, pred_score, attn, all_pred, all_score, gold_score = generate_beam(batch)

  local pred_batch = {}
  local info_batch = {}

  for b = 1, batch.size do
    table.insert(pred_batch, build_target_tokens(pred[b], src_batch[b], attn[b]))

    local info = {}
    info.score = pred_score[b]
    info.n_best = {}

    if gold_score ~= nil then
      info.gold_score = gold_score[b]
    end

    if opt.n_best > 1 then
      for n = 1, opt.n_best do
        info.n_best[n].tokens = build_target_tokens(all_pred[b][n], src_batch[b], attn[b])
        info.n_best[n].score = all_score[b][n]
      end
    end

    table.insert(info_batch, info)
  end

  return pred_batch, info_batch
end

return {
  init = init,
  search = search
}
