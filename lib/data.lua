require('lib.utils.init')

local constants = require 'lib.constants'

--[[ Data management and batch creation.

Batch interface [size]:

  * size: number of sentences in the batch [1]
  * source_length: max length in source batch [1]
  * source_size:  lengths of each source [batch x 1]
  * source_input:  left-padded idx's of source (PPPPPPABCDE) [batch x max]
  * source_input_features: table of source features sequences
  * source_input_rev: right-padded  idx's of source rev (EDCBAPPPPPP) [batch x max]
  * source_input_rev_features: table of reversed source features sequences
  * target_length: max length in source batch [1]
  * target_size: lengths of each source [batch x 1]
  * target_non_zeros: number of non-ignored words in batch [1]
  * target_input: input idx's of target (SABCDEPPPPPP) [batch x max]
  * target_input_features: table of target input features sequences
  * target_output: expected output idx's of target (ABCDESPPPPPP) [batch x max]
  * target_output_features: table of target output features sequences

 TODO: change name of size => maxlen
--]]
local Data = torch.class("Data")

--[[ Return the max_length, sizes, and non-zero count
  of a batch of `seq`s ignoring `ignore` words.
--]]
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

--[[ Initialize a data object given aligned tables of IntTensors `src`
  and `targ`.
--]]
function Data:__init(src_data, targ_data)

  self.src = src_data.words
  self.src_features = src_data.features

  if targ_data ~= nil then
    self.targ = targ_data.words
    self.targ_features = targ_data.features
  end
end

--[[ Setup up the training data to respect `max_batch_size`. ]]
function Data:set_batch_size(max_batch_size)

  self.batch_range = {}
  self.max_source_length = 0
  self.max_target_length = 0

  -- Prepares batches in terms of range within self.src and self.targ.
  local offset = 0
  local batch_size = 1
  local source_length = 0
  local target_length = 0

  for i = 1, #self.src do
    -- Set up the offsets to make same source size batches of the
    -- correct size.
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

    -- Target contains <s> and </s>.
    local target_seq_length = self.targ[i]:size(1) - 1
    target_length = math.max(target_length, target_seq_length)
    self.max_target_length = math.max(self.max_target_length, target_seq_length)
  end
end

--[[ Return number of batches. ]]
function Data:__len__()

  if self.batch_range == nil then
    return 1
  end

  return #self.batch_range
end

--[[ Create a batch object given aligned sent tables `src` and `targ`
  (optional). Data format is shown at the top of the file.
--]]
function Data:get_data(src, src_features, targ, targ_features)

  local batch = {}

  if targ ~= nil then
    assert(#src == #targ, "source and target must have the same batch size")
  end

  batch.size = #src

  batch.source_length, batch.source_size = get_length(src)

  local source_seq = torch.IntTensor(batch.source_length, batch.size):fill(constants.PAD)
  batch.source_input = source_seq:clone()
  batch.source_input_rev = source_seq:clone()

  batch.source_input_features = {}
  batch.source_input_rev_features = {}

  if #src_features > 0 then
    for _ = 1, #src_features[1] do
      table.insert(batch.source_input_features, source_seq:clone())
      table.insert(batch.source_input_rev_features, source_seq:clone())
    end
  end

  if targ ~= nil then
    batch.target_length, batch.target_size, batch.target_non_zeros = get_length(targ, 1)

    local target_seq = torch.IntTensor(batch.target_length, batch.size):fill(constants.PAD)
    batch.target_input = target_seq:clone()
    batch.target_output = target_seq:clone()

    batch.target_input_features = {}
    batch.target_output_features = {}

    if #targ_features > 0 then
      for _ = 1, #targ_features[1] do
        table.insert(batch.target_input_features, target_seq:clone())
        table.insert(batch.target_output_features, target_seq:clone())
      end
    end
  end

  for b = 1, batch.size do
    local source_offset = batch.source_length - batch.source_size[b] + 1
    local source_input = src[b]
    local source_input_rev = src[b]:index(1, torch.linspace(batch.source_size[b], 1, batch.source_size[b]):long())

    -- Source input is left padded [PPPPPPABCDE] .
    batch.source_input[{{source_offset, batch.source_length}, b}]:copy(source_input)
    batch.source_input_pad_left = true

    -- Rev source input is right padded [EDCBAPPPPPP] .
    batch.source_input_rev[{{1, batch.source_size[b]}, b}]:copy(source_input_rev)
    batch.source_input_rev_pad_left = false

    for i = 1, #batch.source_input_features do
      local source_input_features = src_features[b][i]
      local source_input_rev_features = src_features[b][i]:index(1, torch.linspace(batch.source_size[b], 1, batch.source_size[b]):long())

      batch.source_input_features[i][{{source_offset, batch.source_length}, b}]:copy(source_input_features)
      batch.source_input_rev_features[i][{{1, batch.source_size[b]}, b}]:copy(source_input_rev_features)
    end

    if targ ~= nil then
      -- Input: [<s>ABCDE]
      -- Ouput: [ABCDE</s>]
      local target_length = targ[b]:size(1) - 1
      local target_input = targ[b]:narrow(1, 1, target_length)
      local target_output = targ[b]:narrow(1, 2, target_length)

      -- Target is right padded [<S>ABCDEPPPPPP] .
      batch.target_input[{{1, target_length}, b}]:copy(target_input)
      batch.target_output[{{1, target_length}, b}]:copy(target_output)

      for i = 1, #batch.target_input_features do
        local target_input_features = targ_features[b][i]:narrow(1, 1, target_length)
        local target_output_features = targ_features[b][i]:narrow(1, 2, target_length)

        batch.target_input_features[i][{{1, target_length}, b}]:copy(target_input_features)
        batch.target_output_features[i][{{1, target_length}, b}]:copy(target_output_features)
      end
    end
  end

  return batch
end

--[[ Get batch `idx`. If nil make a batch of all the data. ]]
function Data:get_batch(idx)
  if idx == nil or self.batch_range == nil then
    return self:get_data(self.src, self.src_features, self.targ, self.targ_features)
  end

  local range_start = self.batch_range[idx]["begin"]
  local range_end = self.batch_range[idx]["end"]

  local src = {}
  local targ = {}

  local src_features = {}
  local targ_features = {}

  for i = range_start, range_end do
    table.insert(src, self.src[i])
    table.insert(targ, self.targ[i])

    if self.src_features[i] then
      table.insert(src_features, self.src_features[i])
    end

    if self.targ_features[i] then
      table.insert(targ_features, self.targ_features[i])
    end
  end

  return self:get_data(src, src_features, targ, targ_features)
end

return Data
