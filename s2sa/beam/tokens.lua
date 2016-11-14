require 'torch'
local constants = require 's2sa.beam.constants'

local Tokens = {
  replace_unk = false,
  phrase_table = {}
}

function Tokens.init(replace_unk, phrasetable_file)
  Tokens.replace_unk = replace_unk

  if Tokens.replace_unk and phrasetable_file then
    local f = io.open(phrasetable_file,'r')
    if f == nil then
      error('Failed to open file ' .. phrasetable_file)
    end

    local function strip(s)
      return s:gsub("^%s+",""):gsub("%s+$","")
    end

    for line in f:lines() do
      local c = line:split("|||")
      Tokens.phrase_table[strip(c[1])] = c[2]
    end
  end
end

function Tokens.from_sentence(line)
  local tokens = {}
  for word in line:gmatch'([^%s]+)' do
    table.insert(tokens, {value = word})
  end
  return tokens
end

function Tokens.to_sentence(tokens)
  local words = {}
  for _, token in pairs(tokens) do
    table.insert(words, token.value)
  end

  return table.concat(words, ' ')
end

function Tokens.to_words_idx(tokens, word2idx, start_symbol)
  local t = {}
  local u = {}
  if start_symbol == 1 then
    table.insert(t, constants.START)
    table.insert(u, constants.START_WORD)
  end

  for _, token in pairs(tokens) do
    local idx = word2idx[token.value] or constants.UNK
    table.insert(t, idx)
    table.insert(u, token.value)
  end
  if start_symbol == 1 then
    table.insert(t, constants.END)
    table.insert(u, constants.END_WORD)
  end
  return torch.LongTensor(t), u
end

function Tokens.from_words_idx(sent, features, idx2word, idx2feature, source_words, attn, num_target_features)
  local cleaned_tokens = {}
  local cleaned_features = {}
  local cleaned_pos = 0
  local tokens = {}
  local pos = 0

  for i = 2, #sent-1 do -- skip constants.START and constants.END
    local fields = {}
    local token_features = {}
    if sent[i] == constants.UNK and Tokens.replace_unk then
      -- retrieve source word with max attention
      local _, max_index = attn[i]:max(1)
      local s = source_words[max_index[1]]

      if Tokens.phrase_table[s] ~= nil then
        print('Unknown token "' .. s .. '" replaced by source token "' ..Tokens.phrase_table[s] .. '"')
      end
      local r = Tokens.phrase_table[s] or s
      table.insert(fields, r)
    else
      table.insert(fields, idx2word[sent[i]])
    end
    for j = 1, num_target_features do
      local values = {}
      for k = 1, #features[i][j] do
        table.insert(values, idx2feature[j][features[i][j][k]])
      end
      local values_str = table.concat(values, ',')
      table.insert(fields, values_str)
      table.insert(token_features, values)
    end

    local token_value = table.concat(fields, '-|-')
    table.insert(tokens, {
      value = token_value,
      range = {
        begin = pos,
        ['end'] = pos + string.len(token_value)
      },
      attention = attn[i]
    })
    pos = pos + string.len(token_value) + 1

    -- build cleaned tokens and features only if it's necessary
    if num_target_features > 0 then
      table.insert(cleaned_tokens, {
        value = fields[1],
        range = {
          begin = cleaned_pos,
          ['end'] = cleaned_pos + string.len(fields[1])
        },
        attention = attn[i]
      })

      table.insert(cleaned_features, {
        value = token_features,
        range = {
          begin = cleaned_pos,
          ['end'] = cleaned_pos + string.len(fields[1])
        }
      })
      cleaned_pos = cleaned_pos + string.len(fields[1]) + 1
    end
  end

  -- if no features supported => tokens are cleaned
  if num_target_features == 0 then
    cleaned_tokens = tokens
  end

  return tokens, cleaned_tokens, cleaned_features
end

return Tokens
