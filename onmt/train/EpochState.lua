--[[ Class for managing the training process by logging and storing
  the state of the current epoch.
]]
local EpochState = torch.class("EpochState")

--[[ Initialize for epoch `epoch`]]
function EpochState:__init(epoch, numIterations, learningRate)
  self.epoch = epoch
  self.numIterations = numIterations
  self.learningRate = learningRate

  self.globalTimer = torch.Timer()

  self:reset()
end

function EpochState:reset()
  self.trainLoss = 0
  self.sourceWords = 0
  self.targetWordsNonZeros = 0

  self.timer = torch.Timer()
end

--[[ Update training status. Takes `batch` (described in data.lua) and last loss.]]
function EpochState:update(batch, loss)
  self.trainLoss = self.trainLoss + loss
  self.sourceWords = self.sourceWords + batch.size * batch.sourceLength

  if batch.targetNonZeros then
    self.targetWordsNonZeros = self.targetWordsNonZeros + batch.targetNonZeros
  else
    -- If training on monolingual data, loss is normalized by the number of source words.
    self.targetWordsNonZeros = self.sourceWords
  end
end

--[[ Log to status stdout. ]]
function EpochState:log(batchIndex)
  _G.logger:info('Epoch %d ; Iteration %d/%d ; Learning rate %.4f ; Source tokens/s %d ; Perplexity %.2f',
                 self.epoch,
                 batchIndex, self.numIterations,
                 self.learningRate,
                 self.sourceWords / self.timer:time().real,
                 math.exp(self.trainLoss / self.targetWordsNonZeros))

  self:reset()
end

function EpochState:getTime()
  return self.globalTimer:time().real
end

return EpochState
