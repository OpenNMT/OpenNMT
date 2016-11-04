require 'torch'
local constants = require 's2sa.beam.constants'
local utf8 --loaded when using character models only

local Chars = {
  targ_chars_idx = {},
  max_word_length = 0,
  dict = nil,
  use_chars_enc = false,
  use_chars_dec = false
}


local function words_to_chars_idx(word, t)
  t[1] = constants.START
  local i = 2
  for _, char in utf8.next, word do
    char = utf8.char(char)
    local char_idx = Chars.dict:lookup(char) or constants.UNK
    t[i] = char_idx
    i = i+1
    if i >= Chars.max_word_length then
      t[i] = constants.END
      break
    end
  end
  if i < Chars.max_word_length then
    t[i] = constants.END
  end
  return t
end

function Chars.init(opt)
  Chars.use_chars_enc = opt.use_chars_enc
  Chars.use_chars_dec = opt.use_chars_dec

  if Chars.use_chars_enc or Chars.use_chars_dec then
    utf8 = require 'lua-utf8'

    Chars.max_word_length = opt.max_word_length
    Chars.dict = opt.chars_dict

    if Chars.use_chars_dec then
      Chars.targ_chars_idx = torch.LongTensor(#opt.targ_dict.idx_to_label, Chars.max_word_length):fill(constants.PAD)
      for i = 1, #opt.targ_dict.idx_to_label do
        Chars.targ_chars_idx[i] = words_to_chars_idx(opt.targ_dict:lookup(i), Chars.targ_chars_idx[i])
      end
    end
  end
end

function Chars.to_chars_idx(tokens, start_symbol)
  local words = {}
  if start_symbol == 1 then
    table.insert(words, constants.START_WORD)
  end

  for _, token in pairs(tokens) do
    table.insert(words, token.value)
  end
  if start_symbol == 1 then
    table.insert(words, constants.END_WORD)
  end
  local chars = torch.ones(#words, Chars.max_word_length)
  for i = 1, #words do
    chars[i] = words_to_chars_idx(words[i], chars[i])
  end
  return chars, words
end


return Chars
