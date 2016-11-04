require 'torch'
local constants = require 's2sa.beam.constants'
local table_utils = require 's2sa.table_utils'
local stringx = require 'pl.stringx'

local Features = {}

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

return Features
