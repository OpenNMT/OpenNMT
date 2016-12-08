local constants = require('lib.constants')

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
local Batch = torch.class('Batch')

--[[ Create a batch object given aligned sent tables `src` and `targ`
  (optional). Data format is shown at the top of the file.
--]]
function Batch:__init(src, src_features, targ, targ_features)
  if targ ~= nil then
    assert(#src == #targ, "source and target must have the same batch size")
  end

  self.size = #src

  self.source_length, self.source_size = get_length(src)

  local source_seq = torch.IntTensor(self.source_length, self.size):fill(constants.PAD)
  self.source_input = source_seq:clone()
  self.source_input_rev = source_seq:clone()

  self.source_input_features = {}
  self.source_input_rev_features = {}

  if #src_features > 0 then
    for _ = 1, #src_features[1] do
      table.insert(self.source_input_features, source_seq:clone())
      table.insert(self.source_input_rev_features, source_seq:clone())
    end
  end

  if targ ~= nil then
    self.target_length, self.target_size, self.target_non_zeros = get_length(targ, 1)

    local target_seq = torch.IntTensor(self.target_length, self.size):fill(constants.PAD)
    self.target_input = target_seq:clone()
    self.target_output = target_seq:clone()

    self.target_input_features = {}
    self.target_output_features = {}

    if #targ_features > 0 then
      for _ = 1, #targ_features[1] do
        table.insert(self.target_input_features, target_seq:clone())
        table.insert(self.target_output_features, target_seq:clone())
      end
    end
  end

  for b = 1, self.size do
    local source_offset = self.source_length - self.source_size[b] + 1
    local source_input = src[b]
    local source_input_rev = src[b]:index(1, torch.linspace(self.source_size[b], 1, self.source_size[b]):long())

    -- Source input is left padded [PPPPPPABCDE] .
    self.source_input[{{source_offset, self.source_length}, b}]:copy(source_input)
    self.source_input_pad_left = true

    -- Rev source input is right padded [EDCBAPPPPPP] .
    self.source_input_rev[{{1, self.source_size[b]}, b}]:copy(source_input_rev)
    self.source_input_rev_pad_left = false

    for i = 1, #self.source_input_features do
      local source_input_features = src_features[b][i]
      local source_input_rev_features = src_features[b][i]:index(1, torch.linspace(self.source_size[b], 1, self.source_size[b]):long())

      self.source_input_features[i][{{source_offset, self.source_length}, b}]:copy(source_input_features)
      self.source_input_rev_features[i][{{1, self.source_size[b]}, b}]:copy(source_input_rev_features)
    end

    if targ ~= nil then
      -- Input: [<s>ABCDE]
      -- Ouput: [ABCDE</s>]
      local target_length = targ[b]:size(1) - 1
      local target_input = targ[b]:narrow(1, 1, target_length)
      local target_output = targ[b]:narrow(1, 2, target_length)

      -- Target is right padded [<S>ABCDEPPPPPP] .
      self.target_input[{{1, target_length}, b}]:copy(target_input)
      self.target_output[{{1, target_length}, b}]:copy(target_output)

      for i = 1, #self.target_input_features do
        local target_input_features = targ_features[b][i]:narrow(1, 1, target_length)
        local target_output_features = targ_features[b][i]:narrow(1, 2, target_length)

        self.target_input_features[i][{{1, target_length}, b}]:copy(target_input_features)
        self.target_output_features[i][{{1, target_length}, b}]:copy(target_output_features)
      end
    end
  end
end

function Batch:get_source_input(t)
  -- If a regular input, return word id, otherwise a table with features.
  if #self.source_input_features > 0 then
    local inputs = {self.source_input[t]}
    for j = 1, #self.source_input_features do
      table.insert(inputs, self.source_input_features[j][t])
    end
    return inputs
  else
    return self.source_input[t]
  end
end

function Batch:get_target_input(t)
  -- If a regular input, return word id, otherwise a table with features.
  if #self.target_input_features > 0 then
    local inputs = {self.target_input[t]}
    for j = 1, #self.target_input_features do
      table.insert(inputs, self.target_input_features[j][t])
    end
    return inputs
  else
    return self.target_input[t]
  end
end

function Batch:get_target_output(t)
  -- If a regular input, return word id, otherwise a table with features.
  local outputs = { self.target_output[t] }
  for j = 1, #self.target_output_features do
    table.insert(outputs, self.target_output_features[j][t])
  end
  return outputs
end

return Batch
