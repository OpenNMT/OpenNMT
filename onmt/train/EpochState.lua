--[[ Class for managing the training process by logging and storing
  the state of the current epoch.
]]
local EpochState = torch.class("EpochState")

--[[ Initialize for epoch `epoch` and training `status` (current loss)]]
function EpochState:__init(epoch, numIterations, learningRate, lastValidPpl, status)
  self.epoch = epoch
  self.numIterations = numIterations
  self.learningRate = learningRate
  self.lastValidPpl = lastValidPpl

  if status ~= nil then
    self.status = status
  else
    self.status = {}
    self.status.trainNonzeros = 0
    self.status.trainLoss = 0
  end

  self.timer = torch.Timer()
  self.numWordsSource = 0
  self.numWordsTarget = 0
end

--[[ Update training status. Takes `batch` (described in data.lua) and last loss.]]
function EpochState:update(batch, loss)
  self.numWordsSource = self.numWordsSource + batch.size * batch.sourceLength
  if batch.targetLength then
    self.numWordsTarget = self.numWordsTarget + batch.size * batch.targetLength
  end
  self.status.trainLoss = self.status.trainLoss + loss
  if batch.targetNonZeros then
    self.status.trainNonzeros = self.status.trainNonzeros + batch.targetNonZeros
  else
    -- if training on monolingual data - divider is number of source words
    self.status.trainNonzeros = self.status.trainNonzeros + batch.size * batch.sourceLength
  end
end

--[[ Log to status stdout. ]]
function EpochState:log(batchIndex)
  local timeTaken = self:getTime()

  local stats = ''
  stats = stats .. string.format('Epoch %d ; ', self.epoch)
  stats = stats .. string.format('Iteration %d/%d ; ', batchIndex, self.numIterations)
  stats = stats .. string.format('Learning rate %.4f ; ', self.learningRate)
  stats = stats .. string.format('Source tokens/s %d ; ', self.numWordsSource / timeTaken)
  stats = stats .. string.format('Perplexity %.2f', self:getTrainPpl())
  _G.logger:info(stats)
end

function EpochState:getTrainPpl()
  return math.exp(self.status.trainLoss / self.status.trainNonzeros)
end

function EpochState:getTime()
  return self.timer:time().real
end

function EpochState:getStatus()
  return self.status
end

return EpochState
