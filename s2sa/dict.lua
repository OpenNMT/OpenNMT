require 'torch'

local dict = torch.class("dict")

function dict:__init(data)
  self.idx_to_label = {}
  self.label_to_idx = {}
  self.frequencies = {}

  if data ~= nil then
    if type(data) == "string" then -- File to load.
      self:load_file(data)
    end
  end
end

function dict:__len__()
  return #self.idx_to_label
end

function dict:load_file(filename)
  local file = io.open(filename, 'r')

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

function dict:lookup(key)
  if type(key) == "string" then
    return self.label_to_idx[key]
  else
    return self.idx_to_label[key]
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
end

function dict:prune(size)
  if size > #self then
    return self
  end

  local freq = torch.Tensor(self.frequencies)
  local _, idx = torch.sort(freq, 1, true)

  local new_dict = dict.new()

  for i = 1, size do
    new_dict:add(self.idx_to_label[idx[i]])
  end

  return new_dict
end

return dict
