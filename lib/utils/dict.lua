local dict = torch.class("dict")

function dict:__init(data)
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

function dict:__len__()
  return #self.idx_to_label
end

function dict:load_file(filename)
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

function dict:write_file(filename)
  local file = assert(io.open(filename, 'w'))

  for i = 1, #self do
    local label = self.idx_to_label[i]
    file:write(label .. ' ' .. i .. '\n')
  end

  file:close()
end

function dict:lookup(key)
  if type(key) == "string" then
    return self.label_to_idx[key]
  else
    return self.idx_to_label[key]
  end
end

function dict:set_special(special)
  self.special = special
end

function dict:add_special(label, idx)
  idx = self:add(label, idx)
  table.insert(self.special, idx)
end

function dict:add_specials(labels)
  for i = 1, #labels do
    self:add_special(labels[i])
  end
end

function dict:add(label, idx)
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

function dict:prune(size)
  if size >= #self then
    return self
  end

  -- Only keep the `size` most frequent entries.
  local freq = torch.Tensor(self.frequencies)
  local _, idx = torch.sort(freq, 1, true)

  local new_dict = dict.new()

  -- Add special entries in all cases.
  for i = 1, #self.special do
    new_dict:add_special(self.idx_to_label[self.special[i]])
  end

  for i = 1, size do
    new_dict:add(self.idx_to_label[idx[i]])
  end

  return new_dict
end

function dict:convert_to_idx(labels, unk_word, bos_word, eos_word)
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

function dict:convert_to_labels(idx, stop)
  local labels = {}

  for i = 1, #idx do
    table.insert(labels, self:lookup(idx[i]))
    if idx[i] == stop then
      break
    end
  end

  return labels
end

return dict
