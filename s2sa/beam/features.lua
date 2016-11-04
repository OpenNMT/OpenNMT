require 'torch'
local Dict = require 's2sa.dict'
local constants = require 's2sa.beam.constants'
local table_utils = require 's2sa.table_utils'
local stringx = require 'pl.stringx'

local Features = {
  num_source_features = 0,
  num_target_features = 0,
  src_features_dicts = {},
  targ_features_dicts = {},
  source_features_lookup = {},
  target_features_lookup = {},
  hypotheses = {},
  max_hypothesis = {},
  float = false
}

function Features.init(opt)
  Features.num_source_features = opt.num_source_features
  Features.num_target_features = opt.num_target_features
  Features.float = opt.float

  if opt.src_dict and opt.targ_dict then
    Features.src_features_dicts = opt.src_dict
    Features.targ_features_dicts = opt.targ_dict
  elseif opt.dicts_prefix then
    for i = 1, Features.num_source_features do
      table.insert(Features.src_features_dicts, Dict.new(opt.dicts_prefix .. '.source_feature_' .. i .. '.dict'))
    end
    for i = 1, Features.num_target_features do
      table.insert(Features.targ_features_dicts, Dict.new(opt.dicts_prefix .. '.target_feature_' .. i .. '.dict'))
    end
  end

  Features.source_features_lookup = opt.source_features_lookup
  Features.target_features_lookup = opt.target_features_lookup

  if Features.source_features_lookup == nil then
    Features.source_features_lookup = {}
    for _ = 1, Features.num_source_features do
      table.insert(Features.source_features_lookup, false)
    end
    Features.target_features_lookup = {}
    for _ = 1, Features.num_target_features do
      table.insert(Features.target_features_lookup, false)
    end
  end
end

function Features.reset(batch_size, K)
  Features.hypotheses = {}
  Features.max_hypothesis = {}

  for b = 1, batch_size do
    table.insert(Features.max_hypothesis, {})
    table.insert(Features.hypotheses, {})
    for _ = 1, K do
      table.insert(Features.hypotheses[b], {})
    end
  end
end

local function get_feature_embedding(values, features_dict, use_lookup)
  if use_lookup == true then
    local t = torch.Tensor(1)
    local idx = features_dict:lookup(values[1])
    if idx == nil then
      idx = constants.UNK
    end
    t[1] = idx
    return t
  else
    local emb = {}
    for _ = 1, #features_dict.idx_to_label do
      table.insert(emb, 0)
    end
    for i = 1, #values do
      local idx = features_dict:lookup(values[i])
      if idx == nil then
        idx = constants.UNK
      end
      emb[idx] = 1
    end
    return torch.Tensor(emb):view(1,#emb)
  end
end

function Features.to_features_idx(features, features_dicts, use_lookup, start_symbol, decoder)
  local out = {}

  if decoder == 1 then
    table.insert(out, {})
    for j = 1, #features_dicts do
      local emb = get_feature_embedding({constants.UNK_WORD}, features_dicts[j], use_lookup[j])
      table.insert(out[#out], emb)
    end
  end

  if start_symbol == 1 then
    table.insert(out, {})
    for j = 1, #features_dicts do
      local emb = get_feature_embedding({constants.START_WORD}, features_dicts[j], use_lookup[j])
      table.insert(out[#out], emb)
    end
  end

  for i = 1, #features do
    table.insert(out, {})
    for j = 1, #features_dicts do
      local emb = get_feature_embedding(features[i][j], features_dicts[j], use_lookup[j])
      table.insert(out[#out], emb)
    end
  end

  if start_symbol == 1 and decoder == 0 then
    table.insert(out, {})
    for j = 1, #features_dicts do
      local emb = get_feature_embedding({constants.END_WORD}, features_dicts[j], use_lookup[j])
      table.insert(out[#out], emb)
    end
  end

  return out
end

function Features.to_source_features_idx(features, start_symbol, decoder)
  return Features.to_features_idx(features, Features.src_features_dicts, Features.source_features_lookup, start_symbol, decoder)
end

function Features.to_target_features_idx(features, start_symbol, decoder)
  return Features.to_features_idx(features, Features.targ_features_dicts, Features.target_features_lookup, start_symbol, decoder)
end

local function clean_sent(sent)
  local s = stringx.replace(sent, constants.UNK_WORD, '')
  s = stringx.replace(s, constants.START_WORD, '')
  s = stringx.replace(s, constants.END_WORD, '')
  s = stringx.replace(s, constants.START_CHAR, '')
  s = stringx.replace(s, constants.END_CHAR, '')
  return s
end

function Features.extract(tokens)
  local cleaned_tokens = {}
  local features = {}

  for _, entry in pairs(tokens) do
    local field = entry.value:split('%-|%-')
    local word = clean_sent(field[1])
    if string.len(word) > 0 then
      local cleaned_token = table_utils.copy(entry)
      cleaned_token.value = word
      table.insert(cleaned_tokens, cleaned_token)

      if #field > 1 then
        table.insert(features, {})
      end

      for i= 2, #field do
        local values = field[i]:split(',')
        table.insert(features[#features], values)
      end
    end
  end

  return cleaned_tokens, features
end

function Features.get_next_features(batch, n, K)
  local next_ys_features = {}

  if Features.num_target_features > 0 then
    for i = 1, n do
      table.insert(next_ys_features, {})
      for j = 1, Features.num_target_features do
        local t
        if Features.target_features_lookup[j] == true then
          t = torch.LongTensor(batch, K):fill(constants.PAD)
        else
          t = torch.Tensor(batch, K, #Features.targ_features_dicts[j].idx_to_label):zero()
          t[{{}, {}, constants.PAD}]:fill(1)
          if Features.float then
            t = t:float()
          end
        end
        table.insert(next_ys_features[i], t)
      end
    end

    for b = 1, batch do
      for j = 1, Features.num_target_features do
        if Features.target_features_lookup[j] == true then
          next_ys_features[1][j][b][1] = constants.UNK
        else
          next_ys_features[1][j][b][1][constants.UNK] = 1
        end
      end
    end
  end

  return next_ys_features
end

function Features.get_features_input(features, use_lookup, max_len, pad_left)
  local features_input = {}
  local batch = #features

  for i = 1, max_len do
    table.insert(features_input, {})
    for j = 1, #features[1][1] do
      local t
      if use_lookup[j] == true then
        t = torch.LongTensor(batch):fill(constants.PAD)
      else
        t = torch.Tensor(batch, features[1][1][j]:size(2)):zero()
        t[{{}, constants.PAD}]:fill(1)
        if Features.float then
          t = t:float()
        end
      end
      table.insert(features_input[i], t)
    end
  end

  for b = 1, batch do
    for i = 1, #features[b] do
      for j = 1, #features[b][i] do
        local idx
        if pad_left then
          idx = max_len-#features[b]+i
        else
          idx = i
        end
        if use_lookup[j] == true then
          features_input[idx][j][b] = features[b][i][j]
        else
          features_input[idx][j][b]:copy(features[b][i][j])
        end
      end
    end
  end

  return features_input
end

function Features.get_source_features_input(features, max_len, pad_left)
    if Features.num_source_features == 0 then
    return {}
  end
  return Features.get_features_input(features, Features.source_features_lookup, max_len, pad_left)
end

function Features.get_target_features_input(features, max_len, pad_left)
  if Features.num_target_features == 0 then
    return {}
  end
  return Features.get_features_input(features, Features.target_features_lookup, max_len, pad_left)
end

function Features.get_decoder_input(K, remaining_sents)
  local decoder_input_features = {}
  for j = 1, Features.num_target_features do
    local t
    if Features.target_features_lookup[j] == true then
      t = torch.LongTensor(K, remaining_sents)
    else
      t = torch.Tensor(K, remaining_sents, #Features.targ_features_dicts[j].idx_to_label)
      if Features.float then
        t = t:float()
      end
    end
    table.insert(decoder_input_features, t:zero())
  end
  return decoder_input_features
end

function Features.calculate_hypotheses(next_ys_features, out, K, i, b, idx)
  if Features.num_target_features == 0 then
    return
  end

  for k = 1, K do
    table.insert(Features.hypotheses[b][k], {})
    for j = 1, Features.num_target_features do
      local lk, indices = torch.sort(out[1+j][idx][k], true)
      local best = 1
      local hyp = {}
      if Features.target_features_lookup[j] == true then
        next_ys_features[i][j][b][k] = indices[best]
        hyp[1] = indices[best]
      else
        next_ys_features[i][j][b]:copy(out[1+j][idx])
        table.insert(hyp, indices[best])
        for l = best+1, lk:size(1) do
          if lk[best] - lk[l] < 0.05 then
            if indices[l] > constants.END then
              table.insert(hyp, indices[l])
            end
          else
            break
          end
        end
      end
      table.insert(Features.hypotheses[b][k][i-1], hyp)
    end
  end
end

function Features.calculate_max_hypothesis(sent_len, max_k, prev_ks, b)
  if Features.num_target_features == 0 then
    return
  end

  for _ = 1, sent_len-1 do
    table.insert(Features.max_hypothesis[b], {})
  end

  -- follow beam path to build the features sequence
  local k = max_k[b]
  for j = sent_len, 2, -1 do
    k = prev_ks[b][j][k]
    Features.max_hypothesis[b][j-1] = Features.hypotheses[b][k][j-1]
  end
end

return Features
