local unicode = require 'tools.utils.unicode'
local separators = require('tools.utils.separators')

local BPE = torch.class('BPE')

function BPE:__init(opt)
  self.split = string.split
  -- to be able to run the code without torch
  if not self.split then
    self.split = function(t, sep)
      local fields = {}
      local pattern = string.format("([^%s]+)", sep)
      t:gsub(pattern, function(c) fields[#fields+1] = c end)
      return fields
    end
  end
  self.codes = {}
  local f = assert(io.open(opt.bpe_model, "r"))

  self.EOT_marker = opt.EOT_marker or opt.bpe_EOT_marker
  self.BOT_marker = opt.BOT_marker or opt.bpe_EOT_marker
  self.joiner_new = opt.joiner_new
  self.joiner_annotate = opt.joiner_annotate

  local t = f:read("*line")
  local options = self.split(t, ";")
  if (#options == 3 or #options == 4 ) then
    self.prefix = options[1] == "true"
    self.suffix = options[2] == "true"
    self.case_insensitive = options[3] == "true"
    t = f:read("*line")
    if #options == 4 then
      print ("Warning: The 'mode' parameter for tokenization compatibility between train and test has been depreciated, please make sure that the same tokenization parameters are applied while training BPE models and applying them on raw text inputs")
    end
  else
    self.prefix = opt.bpe_mode == "prefix" or opt.bpe_mode == "both"
    self.suffix = opt.bpe_mode == "suffix" or opt.bpe_mode == "both"
    self.case_insensitive = opt.bpe_case_insensitive
  end
  local i = 1

  while not(t == nil) do
    local l = self.split(t, " ")
    if #l == 2 then
      self.codes[t] = i
      i = i + 1
    end
    t=f:read("*line")
  end
end

local function getPairs(word)
  local pairs = {}
  for i = 1, #word-1, 1 do
    table.insert(pairs, word[i] .. ' ' .. word[i+1])
  end
  return pairs
end

local function str2word(l, case_insensitive)
  local word = {}
  for v, c in unicode.utf8_iter(l) do
    if (case_insensitive) then
      local lu, lc = unicode.getLower(v)
        if lu then
          c = lc
        end
    end
    table.insert(word, c)
  end
  return word
end

function BPE:minPair(pairsTable)
  local mintmp = 100000
  local minpair = ''
  for i = 1, #pairsTable, 1 do
    local pair_cur = pairsTable[i]
    if self.codes[pair_cur] then
      local scoretmp = self.codes[pair_cur]
      if (scoretmp < mintmp) then
        mintmp = scoretmp
        minpair = pair_cur
      end
    end
  end
  return minpair
end

function BPE:encode(l)
  local word = str2word(l, self.case_insensitive)
  if #word == 1 then
    word[1] = l
    return word
  end
  if self.prefix then table.insert(word, 1, self.BOT_marker) end
  if self.suffix then table.insert(word, self.EOT_marker) end
  local pairs = getPairs(word)
  while true do
    local bigram = self:minPair(pairs)
    if bigram == '' then break end
    bigram = self.split(bigram, ' ')
    local new_word = {}
    local merge = false
    for _, xx in ipairs(word) do
      if (merge) then
        if xx == bigram[2] then
          table.insert(new_word, bigram[1] .. bigram[2])
          merge = false
        elseif xx == bigram[1] then
          table.insert(new_word, bigram[1])
        else
          table.insert(new_word, bigram[1])
          table.insert(new_word, xx)
          merge = false
        end
      else
        if bigram[1] == xx then
          merge = true
        else
          table.insert(new_word, xx)
        end
      end
    end
    if merge then table.insert(new_word, bigram[1]) end
    word = new_word
    if #word == 1 then
      break
    else
      pairs = getPairs(word)
    end
  end

  if self.suffix then
    if word[#word] == self.EOT_marker then
      table.remove(word, #word)
    elseif string.sub(word[#word],-string.len(self.EOT_marker)) == self.EOT_marker then
      word[#word] = string.sub(word[#word], 1, -string.len(self.EOT_marker)-1)
    end
  end

  if self.prefix then
    if word[1] == self.BOT_marker then
      table.remove(word, 1)
    elseif string.sub(word[1], 1, string.len(self.BOT_marker)) == self.BOT_marker then
      word[1] = string.sub(word[1], string.len(self.BOT_marker)+1)
    end
  end

  if (self.case_insensitive) then
    local tcword = {}
    local prev_idx = 1
    for i = 1, #word do
      local curr_idx = prev_idx+unicode.utf8len(word[i])
      table.insert(tcword, unicode.utf8substr(l, prev_idx, curr_idx - 1))
      prev_idx = curr_idx
    end
    word = tcword
  end
  return word
end

function BPE:segment(tokens, separator)
  local bpeSegment = {}
  for i=1, #tokens do
    local token = tokens[i]
    if token:sub(1, separators.ph_marker_open:len()) == separators.ph_marker_open then
      table.insert(bpeSegment, token)
    else
      local left_sep = false
      local right_sep = false
      if self.joiner_annotate and not self.joiner_new then
        if token:sub(1, #separator) == separator then
          token = token:sub(#separator + 1)
          left_sep = true
        end
        if token:sub(-#separator, -1) == separator then
          token = token:sub(1, -#separator-1)
          right_sep = true
        end
      end
      local bpeTokens = self:encode(token)
      if self.joiner_annotate and not self.joiner_new then
        if left_sep then
          bpeTokens[1] = separator .. bpeTokens[1]
        end
        if right_sep then
          bpeTokens[#bpeTokens] = bpeTokens[#bpeTokens] .. separator
        end
      end
      for j=1, #bpeTokens-1 do
        if self.joiner_annotate then
          if not self.joiner_new then
            table.insert(bpeSegment, bpeTokens[j] .. separator)
          else
            table.insert(bpeSegment, bpeTokens[j])
            table.insert(bpeSegment, separator)
          end
        else
          table.insert(bpeSegment, bpeTokens[j])
        end
      end
      table.insert(bpeSegment, bpeTokens[#bpeTokens])
    end
  end
  return bpeSegment
end

return BPE
