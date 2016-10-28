local data = torch.class("data")
local cuda = require 's2sa.cuda'

function data:__init(d, max_batch_size)
  self.src = d.src
  self.targ = d.targ

  self:build_batches(max_batch_size)
end

function data:build_batches(max_batch_size)
  self.batch_range = {}
  self.source_length = {}
  self.target_length = {}
  self.target_non_zeros = {}
  self.max_source_length = 0
  self.max_target_length = 0

  -- Prepares batches in terms of range within self.src and self.targ
  local size = 0
  local offset = 0
  local batch_size = 1
  local target_length = 0
  local target_non_zeros = 0

  for i = 1, #self.src do
    if batch_size == max_batch_size or self.src[i]:size(1) ~= size then
      if i > 1 then
        table.insert(self.batch_range, { ["begin"] = offset, ["end"] = i - 1 })
        table.insert(self.source_length, size)
        table.insert(self.target_length, target_length)
        table.insert(self.target_non_zeros, target_non_zeros)
      end

      size = self.src[i]:size(1)
      offset = i
      batch_size = 1
      target_length = 0
      target_non_zeros = 0
    else
      batch_size = batch_size + 1
    end

    local target_seq_length = self.targ[i]:size(1) - 1 -- targ contains <s> and </s>

    target_length = math.max(target_length, target_seq_length)
    target_non_zeros = target_non_zeros + target_seq_length

    self.max_source_length = math.max(self.max_source_length, self.src[i]:size(1))
    self.max_target_length = math.max(self.max_target_length, target_seq_length)
  end
end

function data:__len__()
  return #self.batch_range
end

function data:get_batch(idx)
  local batch = {}

  local range_start = self.batch_range[idx]["begin"]
  local range_end = self.batch_range[idx]["end"]

  batch.size = range_end - range_start + 1
  batch.source_length = self.source_length[idx]
  batch.target_length = self.target_length[idx]
  batch.target_non_zeros = self.target_non_zeros[idx]

  batch.source_input = torch.IntTensor(batch.size, batch.source_length):fill(1)
  batch.target_input = torch.IntTensor(batch.size, batch.target_length):fill(1)
  batch.target_output = torch.IntTensor(batch.size, batch.target_length):fill(1)

  for i = range_start, range_end do
    local target_length = self.targ[i]:size(1) - 1 -- targ contains <s> and </s>
    local target_input_view = self.targ[i]:narrow(1, 1, target_length) -- input starts with <s>
    local target_output_view = self.targ[i]:narrow(1, 2, target_length) -- output ends with </s>

    local batch_idx = i - range_start + 1
    batch.source_input[batch_idx]:copy(self.src[i])
    batch.target_input[batch_idx]:narrow(1, 1, target_length):copy(target_input_view)
    batch.target_output[batch_idx]:narrow(1, 1, target_length):copy(target_output_view)
  end

  cuda.convert({batch.source_input, batch.target_input, batch.target_output})

  return batch
end

return data
