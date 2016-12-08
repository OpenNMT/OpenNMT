require('lib.utils.init')

local Batch = require('lib.batch')

--[[ Data management and batch creation. ]]
local Data = torch.class("Data")

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

--[[ Get batch `idx`. If nil make a batch of all the data. ]]
function Data:get_batch(idx)
  if idx == nil or self.batch_range == nil then
    return Batch.new(self.src, self.src_features, self.targ, self.targ_features)
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

  return Batch.new(src, src_features, targ, targ_features)
end

return Data
