local cuda = require 'lib.utils.cuda'
local constants = require 'lib.utils.constants'

local function get_length(seq, ignore)
  local sizes = torch.IntTensor(#seq):zero()
  local max = 0
  local sum = 0

  for i = 1, #seq do
    local len = seq[i]:size(1)
    if ignore ~= nil then
      len = len - ignore
    end
    max = math.max(max, len)
    sum = sum + len
    sizes[i] = len
  end
  return max, sizes, sum
end


local Data = torch.class("Data")

function Data:__init(src, targ)
  self.src = src
  self.targ = targ
end

function Data:set_batch_size(max_batch_size)
  self.batch_range = {}
  self.max_source_length = 0
  self.max_target_length = 0

  -- Prepares batches in terms of range within self.src and self.targ
  local offset = 0
  local batch_size = 1
  local source_length = 0
  local target_length = 0

  for i = 1, #self.src do
    if batch_size == max_batch_size or self.src[i]:size(1) ~= source_length then
      if i > 1 then
        table.insert(self.batch_range, { ["begin"] = offset, ["end"] = i - 1 })
      end

      offset = i
      batch_size = 1
      source_length = self.src[i]:size(1)
      target_length = 0
    else
      batch_size = batch_size + 1
    end

    self.max_source_length = math.max(self.max_source_length, self.src[i]:size(1))

    -- targ contains <s> and </s>
    local target_seq_length = self.targ[i]:size(1) - 1
    target_length = math.max(target_length, target_seq_length)
    self.max_target_length = math.max(self.max_target_length, target_seq_length)
  end
end

function Data:__len__()
  if self.batch_range == nil then
    return 1
  end

  return #self.batch_range
end

function Data:get_data(src, targ)
  local batch = {}

  batch.size = #src

  batch.source_length, batch.source_size = get_length(src)
  batch.source_input = torch.IntTensor(batch.source_length, batch.size):fill(constants.PAD)
  batch.source_input_rev = torch.IntTensor(batch.source_length, batch.size):fill(constants.PAD)

  if targ ~= nil then
    batch.target_length, batch.target_size, batch.target_non_zeros = get_length(targ, 1)
    batch.target_input = torch.IntTensor(batch.target_length, batch.size):fill(constants.PAD)
    batch.target_output = torch.IntTensor(batch.target_length, batch.size):fill(constants.PAD)
  end

  for b = 1, batch.size do
    -- pad source input on the left
    local source_offset = batch.source_length - batch.source_size[b] + 1
    local source_input = src[b]
    local source_input_rev = src[b]:index(1, torch.linspace(batch.source_size[b], 1, batch.source_size[b]):long())
    batch.source_input[{{source_offset, batch.source_length}, b}]:copy(source_input)
    batch.source_input_rev[{{source_offset, batch.source_length}, b}]:copy(source_input_rev)

    if targ ~= nil then
      local target_length = targ[b]:size(1) - 1 -- targ contains <s> and </s>
      local target_input = targ[b]:narrow(1, 1, target_length) -- input starts with <s>
      local target_output = targ[b]:narrow(1, 2, target_length) -- output ends with </s>

      batch.target_input[{{1, target_length}, b}]:copy(target_input)
      batch.target_output[{{1, target_length}, b}]:copy(target_output)
    end
  end

  batch.source_input = cuda.convert(batch.source_input)
  batch.source_input_rev = cuda.convert(batch.source_input_rev)

  if targ ~= nil then
    batch.target_input = cuda.convert(batch.target_input)
    batch.target_output = cuda.convert(batch.target_output)
  end

  return batch
end

function Data:get_batch(idx)
  if idx == nil or self.batch_range == nil then
    return self:get_data(self.src, self.targ)
  end

  local range_start = self.batch_range[idx]["begin"]
  local range_end = self.batch_range[idx]["end"]

  local src = {}
  local targ = {}

  for i = range_start, range_end do
    table.insert(src, self.src[i])
    table.insert(targ, self.targ[i])
  end

  return self:get_data(src, targ)
end

return Data
