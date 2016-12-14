--[[ Class for managing the training process by logging and storing
  the state of the current epoch.
]]
local EpochState = torch.class("EpochState")

--[[ Initialize for epoch `epoch` and training `status` (current loss)]]
function EpochState:__init(epoch, num_iterations, learning_rate, last_valid_ppl, status)
  self.epoch = epoch
  self.num_iterations = num_iterations
  self.learning_rate = learning_rate
  self.last_valid_ppl = last_valid_ppl

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

  self.minFreeMemory = 100000000000
end

--[[ Update training status. Takes `batch` (described in data.lua) and last losses.]]
function EpochState:update(batches, losses)
  for i = 1,#batches do
    self.num_words_source = self.num_words_source + batches[i].size * batches[i].source_length
    self.num_words_target = self.num_words_target + batches[i].size * batches[i].target_length
    self.status.train_loss = self.status.train_loss + losses[i]
    self.status.train_nonzeros = self.status.train_nonzeros + batches[i].target_non_zeros
  end
end

--[[ Log to status stdout. ]]
function EpochState:log(batch_index, json)
  if json then
    local freeMemory = onmt.utils.Cuda.freeMemory()
    if freeMemory < self.minFreeMemory then
      self.minFreeMemory = freeMemory
    end

    local obj = {
      time = os.time(),
      epoch = self.epoch,
      iteration = batch_index,
      totalIterations = self.num_iterations,
      learningRate = self.learning_rate,
      trainingPerplexity = self:get_train_ppl(),
      freeMemory = freeMemory,
      lastValidationPerplexity = self.last_valid_ppl,
      processedTokens = {
        source = self.num_words_source,
        target = self.num_words_target
      }
    }

    onmt.utils.Log.logJson(obj)
  else
    local time_taken = self:get_time()

    local stats = ''
    stats = stats .. string.format('Epoch %d ; ', self.epoch)
    stats = stats .. string.format('Iteration %d/%d ; ', batch_index, self.num_iterations)
    stats = stats .. string.format('Learning rate %.4f ; ', self.learning_rate)
    stats = stats .. string.format('Source tokens/s %d ; ', self.num_words_source / time_taken)
    stats = stats .. string.format('Perplexity %.2f', self:get_train_ppl())
    print(stats)
  end
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

function EpochState:get_min_freememory()
  return self.minFreeMemory
end

return EpochState
