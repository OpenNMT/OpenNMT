local unicode = require './unicode'

local BPE = torch.class('BPE')

function BPE:__init(codesfile_path)
  local codes = {}
  local f = assert(io.open(codesfile_path, "r"))
  local t=f:read("*line")
  local i=1

  while not(t == nil) do
    local l=string.split(t," ")
    if (#l==2) then
      codes[t]=i
      i=i+1
    end
    t=f:read("*line")
  end
  self.codes = codes
end

local function getPairs(word)
  local pairs = {}
  for i=1, #word-1, 1 do
    table.insert(pairs, word[i]..' '..word[i+1])
  end
  return pairs
end

local function str2word(l)
  local word = {}
  for v, c in unicode.utf8_iter(l) do
    table.insert(word, c)
  end
  table.insert(word, '</w>')
  return word
end

function BPE:minPair(pairsTable)
  local mintmp = 100000
  local minpair = ''
  for i=1, #pairsTable, 1 do
    local pair_cur = pairsTable[i]
    if (self.codes[pair_cur] ~= nil) then
      scoretmp = self.codes[pair_cur]
      if (scoretmp < mintmp) then
        mintmp = scoretmp
        minpair = pair_cur
      end
    end
  end
  return minpair
end

function BPE:encode(l)
  local nextv, nextc = unicode._utf8_to_cp(l, 1)
  if #l <= #nextc then
    local w = {}
    table.insert(w, l)
    return w
  end
  local word = str2word(l)
  local pairs = getPairs(word)
  while (true) do
    local bigram = self:minPair(pairs)
    if (bigram == '') then
      break
    end
    bigram = string.split(bigram," ")
    local new_word = {}
    local merge = false
    for ii, xx in ipairs(word) do
      if (merge) then
        if (bigram[2] == xx) then
          table.insert(new_word, bigram[1]..bigram[2])
        else
          table.insert(new_word, bigram[1])
	  end
        merge = false
      else
        if (bigram[1] == xx ) then
          merge = true
        else
          table.insert(new_word, xx)
        end
      end
    end
    word = new_word
    if #word == 1 then
      break
    else
      pairs = getPairs(word)
    end
  end
  if word[#word] == '</w>' then
    table.remove(word, #word)
  elseif (string.sub(word[#word],-string.len('</w>'))=='</w>') then
    word[#word] = string.sub(word[#word], 1, -string.len('</w>')-1)
  end
  return word
end

return BPE