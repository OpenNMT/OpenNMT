local Dict = torch.class("Dict")

function Dict:__init(data)
  self.idx_to_label = {}
  self.label_to_idx = {}
  self.frequencies = {}

  -- Special entries will not be pruned.
  self.special = {}

  if data ~= nil then
    if type(data) == "string" then -- File to load.
      self:load_file(data)
    else
      self:add_specials(data)
    end
  end
end

--[[ Return the number of entries in the dictionary. ]]
function Dict:__len__()
  return #self.idx_to_label
end

--[[ Load entries from a file. ]]
function Dict:load_file(filename)
  local file = assert(io.open(filename, 'r'))

  for line in file:lines() do
    local fields = {}
    for w in line:gmatch'([^%s]+)' do
      table.insert(fields, w)
    end

    local label = fields[1]
    local idx = tonumber(fields[2])

    self:add(label, idx)
  end

  file:close()
end

--[[ Write entries to a file. ]]
function Dict:write_file(filename)
  local file = assert(io.open(filename, 'w'))

  for i = 1, #self do
    local label = self.idx_to_label[i]
    file:write(label .. ' ' .. i .. '\n')
  end

  file:close()
end

--[[ Lookup `key` in the dictionary: it can be an index or a string. ]]
function Dict:lookup(key)
  if type(key) == "string" then
    return self.label_to_idx[key]
  else
    return self.idx_to_label[key]
  end
end

--[[ Mark this `label` and `idx` as special (i.e. will not be pruned). ]]
function Dict:add_special(label, idx)
  idx = self:add(label, idx)
  table.insert(self.special, idx)
end

--[[ Mark all labels in `labels` as specials (i.e. will not be pruned). ]]
function Dict:add_specials(labels)
  for i = 1, #labels do
    self:add_special(labels[i])
  end
end

--[[ Add `label` in the dictionary. Use `idx` as its index if given. ]]
function Dict:add(label, idx)
  if idx ~= nil then
    self.idx_to_label[idx] = label
    self.label_to_idx[label] = idx
  else
    idx = self.label_to_idx[label]
    if idx == nil then
      idx = #self.idx_to_label + 1
      self.idx_to_label[idx] = label
      self.label_to_idx[label] = idx
    end
  end

  if self.frequencies[idx] == nil then
    self.frequencies[idx] = 1
  else
    self.frequencies[idx] = self.frequencies[idx] + 1
  end

  return idx
end

--[[ Return a new dictionary with the `size` most frequent entries. ]]
function Dict:prune(size)
  if size >= #self then
    return self
  end

  -- Only keep the `size` most frequent entries.
  local freq = torch.Tensor(self.frequencies)
  local _, idx = torch.sort(freq, 1, true)

  local new_dict = Dict.new()

  -- Add special entries in all cases.
  for i = 1, #self.special do
    new_dict:add_special(self.idx_to_label[self.special[i]])
  end

  for i = 1, size do
    new_dict:add(self.idx_to_label[idx[i]])
  end

  return new_dict
end

--[[
  Convert `labels` to indices. Use `unk_word` if not found.
  Optionally insert `bos_word` at the beginning and `eos_word` at the end.
]]
function Dict:convert_to_idx(labels, unk_word, bos_word, eos_word)
  local vec = {}

  if bos_word ~= nil then
    table.insert(vec, self:lookup(bos_word))
  end

  for i = 1, #labels do
    local idx = self:lookup(labels[i])
    if idx == nil then
      idx = self:lookup(unk_word)
    end
    table.insert(vec, idx)
  end

  if eos_word ~= nil then
    table.insert(vec, self:lookup(eos_word))
  end

  return torch.IntTensor(vec)
end

--[[ Convert `idx` to labels. If index `stop` is reached, convert it and return. ]]
function Dict:convert_to_labels(idx, stop)
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
