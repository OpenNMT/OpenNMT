require('nngraph')

--[[ Sequencer is the base class for encoder and decoder models.
  Main task is to manage `self.net(t)`, the unrolled network
  used during training.

     :net(1) => :net(2) => ... => :net(n-1) => :net(n)

--]]
local Sequencer, parent = torch.class('onmt.Sequencer', 'nn.Container')

--[[
Parameters:

  * `network` - recurrent step template.
  * `rnnClass` - (optional) the class name of the underlying RNN (e.g. 'onmt.LSTM')
--]]
function Sequencer:__init(network, rnnClass)
  parent.__init(self)

  self.network = network
  self.rnnClass = rnnClass
  self:add(self.network)

  self.networkClones = {}
end

function Sequencer:_sharedClone()
  local clone = self.network:clone('weight', 'gradWeight', 'bias', 'gradBias')

  -- Manually share word embeddings if they are fixed as they are not declared as parameters.
  local wordEmb
  local rnn
  local crnn

  self.network:apply(function(m)
    if m.fix then
      wordEmb = m
    end
    if m.recurrents then
      rnn = m
    end
  end)

  clone:apply(function(m)
    if wordEmb and m.fix then
      m:share(wordEmb, 'weight')
    end
    if rnn and m.recurrents then
      m:shareRecurrents(rnn)
      crnn = m
    end
  end)

  function clone:setTimeStep(t)
    if crnn ~= nil and crnn.setTimeStep ~= nil then
      crnn:setTimeStep(t)
    end
  end

  -- Share intermediate tensors if defined.
  if self.networkClones[1] then
    local sharedTensors = {}

    self.networkClones[1]:apply(function(m)
      if m.gradInputSharedIdx then
        sharedTensors[m.gradInputSharedIdx] = m.gradInput
      end
      if m.outputSharedIdx then
        sharedTensors[m.outputSharedIdx] = m.output
      end
    end)

    clone:apply(function(m)
      if m.gradInputSharedIdx then
        m.gradInput = sharedTensors[m.gradInputSharedIdx]
      end
      if m.outputSharedIdx then
        m.output = sharedTensors[m.outputSharedIdx]
      end
    end)
  end

  collectgarbage()

  return clone
end

--[[Get access to the recurrent unit at a timestep.

Parameters:
  * `t` - timestep.

Returns: The raw network clone at timestep t.
  When `evaluate()` has been called, cheat and return t=1.
]]
function Sequencer:net(t)
  if self.train then
    -- In train mode, the network has to be cloned to remember intermediate
    -- outputs for each timestep and to allow backpropagation through time.
    if self.networkClones[t] == nil then
      local clone = self:_sharedClone()
      clone:training()
      self.networkClones[t] = clone
    end
    self.networkClones[t]:setTimeStep(t)
    return self.networkClones[t]
  else
    if #self.networkClones > 0 then
      self.networkClones[1]:setTimeStep(t)
      return self.networkClones[1]
    else
      self.network:setTimeStep(t)
      return self.network
    end
  end
end

--[[ Move the network to train mode. ]]
function Sequencer:training()
  parent.training(self)

  if #self.networkClones > 0 then
    -- Only first clone can be used for evaluation.
    self.networkClones[1]:training()
  end
end

--[[ Move the network to evaluation mode. ]]
function Sequencer:evaluate()
  parent.evaluate(self)

  if #self.networkClones > 0 then
    self.networkClones[1]:evaluate()
  end
end
