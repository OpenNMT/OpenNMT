require('../onmt')
require('../utils')

local Data = require('lib.data')
local constants = require('lib.constants')

local checkpoint = nil
local models = {}
local dicts = {}
local opt = {}

local phrase_table

local function init(args)
  opt = args
  utils.Cuda.init(opt)

  print('Loading ' .. opt.model .. '...')
  checkpoint = torch.load(opt.model)

  local encoder_args = {
    num_layers = checkpoint.options.num_layers,
    rnn_size = checkpoint.options.rnn_size,
    mask_padding = true
  }

  local decoder_args = {
    rnn_size = checkpoint.options.rnn_size,
    num_layers = checkpoint.options.num_layers,
    input_feed = checkpoint.options.input_feed == 1,
    mask_padding = true
  }

  if checkpoint.options.brnn then
    models.encoder = onmt.BiEncoder.new(encoder_args, checkpoint.options.brnn_merge, checkpoint.nets.encoder, checkpoint.nets.encoder_bwd)
  else
    models.encoder = onmt.Encoder.new(encoder_args, checkpoint.nets.encoder)
  end

  models.decoder = onmt.Decoder.new(decoder_args, checkpoint.nets.decoder, checkpoint.nets.generator)

  models.encoder:evaluate()
  models.decoder:evaluate()

  utils.Cuda.convert(models.encoder)
  utils.Cuda.convert(models.decoder)

  dicts = checkpoint.dicts

  if opt.srctarg_dict:len() > 0 then
    phrase_table = eval.PhraseTable.new(opt.srctarg_dict)
  end
end

local function build_data(src_batch, src_features_batch, gold_batch, gold_features_batch)
  local src_data = {}
  src_data.words = {}
  src_data.features = {}

  local targ_data
  if gold_batch ~= nil then
    targ_data = {}
    targ_data.words = {}
    targ_data.features = {}
  end

  for b = 1, #src_batch do
    table.insert(src_data.words, dicts.src.words:convert_to_idx(src_batch[b], constants.UNK_WORD))

    if #dicts.src.features > 0 then
      table.insert(src_data.features,
                   utils.Features.generateSource(dicts.src.features, src_features_batch[b]))
    end

    if targ_data ~= nil then
      table.insert(targ_data.words,
                   dicts.targ.words:convert_to_idx(gold_batch[b],
                                                   constants.UNK_WORD,
                                                   constants.BOS_WORD,
                                                   constants.EOS_WORD))

      if #dicts.targ.features > 0 then
        table.insert(targ_data.features,
                     utils.Features.generateTarget(dicts.targ.features, gold_features_batch[b]))
      end
    end
  end

  return Data.new(src_data, targ_data)
end

local function build_target_tokens(pred, pred_feats, src, attn)
  local tokens = dicts.targ.words:convert_to_labels(pred, constants.EOS)

  -- Always ignore last token to stay consistent, even it may not be EOS.
  table.remove(tokens)

  if opt.replace_unk then
    for i = 1, #tokens do
      if tokens[i] == constants.UNK_WORD then
        local _, max_index = attn[i]:max(1)
        local source = src[max_index[1]]

        if phrase_table and phrase_table:contains(source) then
          tokens[i] = phrase_table:lookup(source)
        else
          tokens[i] = source
        end
      end
    end
  end

  if pred_feats ~= nil then
    tokens = utils.Features.annotate(tokens, pred_feats, dicts.targ.features)
  end

  return tokens
end

local function translate_batch(batch)
  -- Forget previous padding module on the decoder.
  models.decoder:reset()

  local enc_states, context = models.encoder:forward(batch)

  local gold_score
  if batch.target_input ~= nil then
    if batch.size > 1 then
      models.decoder:reset(batch.source_size, batch.source_length)
    end
    gold_score = models.decoder:compute_score(batch, enc_states, context)
  end

  -- Expand tensors for each beam.
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

  -- As finished sentences are removed from the batch, this table maps the batches
  -- to their index within the remaining sentences.
  local batch_idx = {}

  local beam = {}

  for b = 1, batch.size do
    table.insert(beam, eval.Beam.new(opt.beam, #dicts.targ.features))
    table.insert(batch_idx, b)
  end

  local i = 1

  local dec_out
  local dec_states = enc_states

  while remaining_sents > 0 and i < opt.max_sent_l do
    i = i + 1

    -- Prepare decoder input.
    local input = torch.IntTensor(opt.beam, remaining_sents)
    local input_features = {}
    local source_sizes = torch.IntTensor(remaining_sents)

    for b = 1, batch.size do
      if not beam[b].done then
        local idx = batch_idx[b]
        source_sizes[idx] = batch.source_size[b]

        -- Get current state of the beam search.
        local word_state, features_state = beam[b]:get_current_state()
        input[{{}, idx}]:copy(word_state)

        for j = 1, #dicts.targ.features do
          if input_features[j] == nil then
            input_features[j] = torch.IntTensor(opt.beam, remaining_sents)
          end
          input_features[j][{{}, idx}]:copy(features_state[j])
        end
      end
    end

    input = input:view(opt.beam * remaining_sents)
    for j = 1, #dicts.targ.features do
      input_features[j] = input_features[j]:view(opt.beam * remaining_sents)
    end

    if batch.size > 1 then
      models.decoder:reset(source_sizes, batch.source_length, opt.beam)
    end

    dec_out, dec_states = models.decoder:forward_one(input, input_features, dec_states, context, dec_out)

    local out = models.decoder.generator:forward(dec_out)

    for j = 1, #out do
      out[j] = out[j]:view(opt.beam, remaining_sents, out[j]:size(2)):transpose(1, 2):contiguous()
    end
    local word_lk = out[1]

    local softmax_out = models.decoder.softmax_attn.output:view(opt.beam, remaining_sents, -1)
    local new_remaining_sents = remaining_sents

    for b = 1, batch.size do
      if not beam[b].done then
        local idx = batch_idx[b]

        local feats_lk = {}
        for j = 1, #dicts.targ.features do
          table.insert(feats_lk, out[j + 1][idx])
        end

        if beam[b]:advance(word_lk[idx], feats_lk, softmax_out[{{}, idx}]) then
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
      -- Update sentence indices within the batch and mark sentences to keep.
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

      -- Update rnn states and context.
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

      -- The `index()` method allocates a new storage so clean the previous ones to
      -- keep a stable memory usage.
      collectgarbage()
    end

    remaining_sents = new_remaining_sents
  end

  local all_hyp = {}
  local all_feats = {}
  local all_attn = {}
  local all_scores = {}

  for b = 1, batch.size do
    local scores, ks = beam[b]:sort_best()

    local hyp_batch = {}
    local feats_batch = {}
    local attn_batch = {}
    local scores_batch = {}

    for n = 1, opt.n_best do
      local hyp, feats, attn = beam[b]:get_hyp(ks[n])

      -- remove unnecessary values from the attention vectors
      for j = 1, #attn do
        local size = batch.source_size[b]
        attn[j] = attn[j]:narrow(1, batch.source_length - size + 1, size)
      end

      table.insert(hyp_batch, hyp)
      if #feats > 0 then
        table.insert(feats_batch, feats)
      end
      table.insert(attn_batch, attn)
      table.insert(scores_batch, scores[n])
    end

    table.insert(all_hyp, hyp_batch)
    table.insert(all_feats, feats_batch)
    table.insert(all_attn, attn_batch)
    table.insert(all_scores, scores_batch)
  end

  return all_hyp, all_feats, all_scores, all_attn, gold_score
end

local function translate(src_batch, src_features_batch, gold_batch, gold_features_batch)
  local data = build_data(src_batch, src_features_batch, gold_batch, gold_features_batch)
  local batch = data:get_batch()

  local pred, pred_feats, pred_score, attn, gold_score = translate_batch(batch)

  local pred_batch = {}
  local info_batch = {}

  for b = 1, batch.size do
    table.insert(pred_batch, build_target_tokens(pred[b][1], pred_feats[b][1], src_batch[b], attn[b][1]))

    local info = {}
    info.score = pred_score[b][1]
    info.n_best = {}

    if gold_score ~= nil then
      info.gold_score = gold_score[b]
    end

    if opt.n_best > 1 then
      for n = 1, opt.n_best do
        info.n_best[n] = {}
        info.n_best[n].tokens = build_target_tokens(pred[b][n], pred_feats[b][n], src_batch[b], attn[b][n])
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
