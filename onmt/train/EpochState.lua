--[[ Class for managing the training process by logging and storing
  the state of the current epoch.
]]
local EpochState = torch.class('EpochState')

--[[ Initialize for epoch `epoch`]]
function EpochState:__init(epoch, startIterations, numIterations, learningRate)
  self.epoch = epoch
  self.iterations = startIterations - 1
  self.numIterations = numIterations
  self.learningRate = learningRate

  self.globalTimer = torch.Timer()

  self:reset()
end

function EpochState:reset()
  self.trainLoss = 0
  self.sourceWords = 0
  self.targetWords = 0

  self.timer = torch.Timer()
end

--[[ Update training status. Takes `batch` (described in data.lua) and last loss.]]
function EpochState:update(model, batch, loss)
  self.iterations = self.iterations + 1
  self.trainLoss = self.trainLoss + loss
  self.sourceWords = self.sourceWords + model:getInputLabelsCount(batch)
  self.targetWords = self.targetWords + model:getOutputLabelsCount(batch)
end

--[[ Log to status stdout. ]]
function EpochState:log(iteration)
  _G.logger:info('Epoch %d ; Iteration %d/%d ; Learning rate %.4f ; Source tokens/s %d ; Perplexity %.2f',
                 self.epoch,
                 iteration or self.iterations, self.numIterations,
                 self.learningRate,
                 self.sourceWords / self.timer:time().real,
                 math.exp(self.trainLoss / self.targetWords))

  self:reset()
end

function EpochState:getTime()
  return self.globalTimer:time().real
end

function EpochState:getAvgPpl()
  return math.exp(self.trainLoss / self.targetWords)
end
return EpochState
