require 'torch'

--[[ Class for managing the training process by logging and storing
  the state of the current epoch.
]]
local EpochState = torch.class("EpochState")

--[[ Initialize for epoch `epoch` and training `status` (current loss)]]
function EpochState:__init(epoch, status)
  self.epoch = epoch

  if status ~= nil then
    self.status = status
  else
    self.status = {}
    self.status.train_nonzeros = 0
    self.status.train_loss = 0
  end

  self.timer = torch.Timer()
  self.num_words_source = 0
  self.num_words_target = 0

  print('')
end

--[[ Update training status. Takes `batch` (described in data.lua) and last losses.]]
function EpochState:update(batches, losses)
  for i = 1,#batches do
    self.num_words_source = self.num_words_source + batches[i].size * batches[i].source_length
    self.num_words_target = self.num_words_target + batches[i].size * batches[i].target_length
    self.status.train_loss = self.status.train_loss + losses[i] * batches[i].size
  end
  -- each batch is containing the complete number of non zeros target words
  self.status.train_nonzeros = self.status.train_nonzeros + batches[1].target_non_zeros
end

--[[ Log to status stdout.
  TODO: these args shouldn't need to be passed in each time. ]]
function EpochState:log(batch_index, data_size, learning_rate)
  local time_taken = self:get_time()
  local stats = ''
  local freeMemory = utils.Cuda.freeMemory()
  stats = stats .. string.format('Epoch %d ; Batch %d/%d ; LR %.4f ; ',
                                 self.epoch, batch_index, data_size, learning_rate)
  stats = stats .. string.format('Throughput %d/%d/%d total/src/targ tokens/sec ; ',
                                 (self.num_words_target + self.num_words_source) / time_taken,
                                 self.num_words_source / time_taken,
                                 self.num_words_target / time_taken)
  stats = stats .. string.format('PPL %.2f ; Free mem %d', self:get_train_ppl(), freeMemory)

  print(stats)
end

function EpochState:get_train_ppl()
  return math.exp(self.status.train_loss / self.status.train_nonzeros)
end

function EpochState:get_time()
  return self.timer:time().real
end

function EpochState:get_status()
  return self.status
end

return EpochState
