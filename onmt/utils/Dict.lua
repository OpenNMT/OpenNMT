local Dict = torch.class("Dict")

function Dict:__init(data)
  self.idxToLabel = {}
  self.labelToIdx = {}
  self.frequencies = {}
  self.freqTensor = nil

  -- Special entries will not be pruned.
  self.special = {}

  if data ~= nil then
    if type(data) == "string" then -- File to load.
      self:loadFile(data)
    else
      self:addSpecials(data)
    end
  end
end

--[[ Return the number of entries in the dictionary. ]]
function Dict:size()
  return #self.idxToLabel
end

--[[ Load entries from a file. ]]
function Dict:loadFile(filename)
  local reader = onmt.utils.FileReader.new(filename)

  while true do
    local fields = reader:next()

    if not fields then
      break
    end

    local label = fields[1]
    local idx = tonumber(fields[2])

    self:add(label, idx)
  end

  reader:close()
end

--[[ Write entries to a file. ]]
function Dict:writeFile(filename)
  local file = assert(io.open(filename, 'w'))

  for i = 1, self:size() do
    local label = self.idxToLabel[i]
    if self.frequencies then
      file:write(label .. ' ' .. i .. ' ' .. (self.frequencies[i] or 0) .. '\n')
    elseif self.freqTensor then
      file:write(label .. ' ' .. i .. ' ' .. self.freqTensor[i] .. '\n')
    else
      file:write(label .. ' ' .. i .. '\n')
    end
  end

  file:close()
end

--[[ Drop or serialize the frequency tensor. ]]
function Dict:prepFrequency(keep)
  if not keep then
    self.freqTensor = nil
  else
    self.freqTensor = torch.Tensor(self.frequencies)
  end
  self.frequencies = nil
end

--[[ Lookup `key` in the dictionary: it can be an index or a string. ]]
function Dict:lookup(key)
  if type(key) == "string" then
    return self.labelToIdx[key]
  else
    return self.idxToLabel[key]
  end
end

--[[ Mark this `label` and `idx` as special (i.e. will not be pruned). ]]
function Dict:addSpecial(label, idx, frequency)
  idx = self:add(label, idx, frequency)
  table.insert(self.special, idx)
end

--[[ Mark all labels in `labels` as specials (i.e. will not be pruned). ]]
function Dict:addSpecials(labels)
  for i = 1, #labels do
    self:addSpecial(labels[i], nil, 0)
  end
end

--[[ Set the frequency of a vocab. ]]
function Dict:setFrequency(label, frequency)
  local idx = self.labelToIdx[label]
  assert(idx)
  self.frequencies[idx] = frequency
end

--[[ Add `label` in the dictionary. Use `idx` as its index if given. ]]
function Dict:add(label, idx, frequency)
  if not frequency then
    frequency = 1
  end
  if idx ~= nil then
    self.idxToLabel[idx] = label
    self.labelToIdx[label] = idx
  else
    idx = self.labelToIdx[label]
    if idx == nil then
      idx = #self.idxToLabel + 1
      self.idxToLabel[idx] = label
      self.labelToIdx[label] = idx
    end
  end

  if self.frequencies[idx] == nil then
    self.frequencies[idx] = frequency
  else
    self.frequencies[idx] = self.frequencies[idx] + frequency
  end

  return idx
end

--[[ Return a new dictionary with the `size` most frequent entries. ]]
function Dict:prune(size)
  if size >= self:size() then
    return self
  end

  -- Only keep the `size` most frequent entries.
  local freq = torch.Tensor(self.frequencies)
  local sortedFreq, idx = torch.sort(freq, 1, true)

  local newDict = Dict.new()

  -- Add special entries in all cases.
  for i = 1, #self.special do
    local thevocab = self.idxToLabel[self.special[i]]
    local thefreq = self.frequencies[self.special[i]]
    if thevocab == onmt.Constants.UNK_WORD then
      thefreq = sortedFreq:narrow(1, size+1, sortedFreq:size()[1]-size):sum()
    end
    newDict:addSpecial(thevocab, nil, thefreq)
  end

  for i = 1, size do
    newDict:add(self.idxToLabel[idx[i]], nil, self.frequencies[idx[i]])
  end

  return newDict
end

--[[ Return a new dictionary with entries appearing at least `minFrequency` times. ]]
function Dict:pruneByMinFrequency(minFrequency)
  if minFrequency < 2 then
    return self
  end

  local freq = torch.Tensor(self.frequencies)
  local sortedFreq, idx = torch.sort(freq, 1, true)

  local newDict = Dict.new()

  -- Add special entries in all cases.
  for i = 1, #self.special do
    local thevocab = self.idxToLabel[self.special[i]]
    local thefreq = self.frequencies[self.special[i]]
    newDict:addSpecial(thevocab, nil, thefreq)
  end

  for i = 1, self:size() do
    if sortedFreq[i] < minFrequency then
      newDict:setFrequency(onmt.Constants.UNK_WORD, sortedFreq:narrow(1, i, sortedFreq:size()[1]-i):sum())
      break
    end
    newDict:add(self.idxToLabel[idx[i]], nil, sortedFreq[i])
  end

  return newDict
end

--[[ Add frequency to current dictionary from provided dictionary ]]
function Dict:getFrequencies(dict)
  local newDict = Dict.new()

  for i = 1, dict:size() do
    local token = dict:idxToLabel(i)
    local idx = self:labelToIdx(token)
    local frequency = 0
    if idx then
      frequency = self.frequencies[idx]
    end
    newDict:add(token, i)
    newDict.frequencies[i] = frequency
  end

  return newDict
end

--[[
  Convert `labels` to indices. Use `unkWord` if not found.
  Optionally insert `bosWord` at the beginning and `eosWord` at the end.
]]
function Dict:convertToIdx(labels, unkWord, bosWord, eosWord)
  local vec = {}

  if bosWord ~= nil then
    table.insert(vec, self:lookup(bosWord))
  end

  for i = 1, #labels do
    local idx = self:lookup(labels[i])
    if idx == nil then
      idx = self:lookup(unkWord)
    end
    table.insert(vec, idx)
  end

  if eosWord ~= nil then
    table.insert(vec, self:lookup(eosWord))
  end

  return torch.IntTensor(vec)
end

--[[ Convert `idx` to labels. If index `stop` is reached, convert it and return. ]]
function Dict:convertToLabels(idx, stop)
  local labels = {}

  for i = 1, #idx do
    table.insert(labels, self:lookup(idx[i]))
    if idx[i] == stop then
      break
    end
  end

  return labels
end

return Dict
