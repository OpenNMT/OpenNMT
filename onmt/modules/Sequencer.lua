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
--]]
function Sequencer:__init(network)
  parent.__init(self)

  self.network = network
  self:add(self.network)

  self.networkClones = {}
end

function Sequencer:_sharedClone()
  local clone = self.network:clone('weight', 'gradWeight', 'bias', 'gradBias', 'fullWeight', 'fullBias', 'sharedNoise', 'noiseInit')

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
  if self.train and t then
    -- In train mode, the network has to be cloned to remember intermediate
    -- outputs for each timestep and to allow backpropagation through time.
    while #self.networkClones < t do
      table.insert(self.networkClones, self:_sharedClone())
    end
    return self.networkClones[t]
  elseif #self.networkClones > 0 then
    return self.networkClones[1]
  else
    return self.network
  end
end

--[[Return the id of the clone to use for timestep t or 0 if not using clones]]
function Sequencer:cloneId(t)
  if self.train and t then
    return t
  elseif #self.networkClones > 0 then
    return 1
  else
    return 0
  end
end


--[[ Move the network to train mode. ]]
function Sequencer:training()
  parent.training(self)

  if #self.networkClones > 0 then
    -- Only first clone was used for evaluation.
    self.networkClones[1]:training()
  end
end

--[[ Move the network to evaluation mode. ]]
function Sequencer:evaluate()
  parent.evaluate(self)

  if #self.networkClones > 0 then
    -- We only use the first clone for evaluation.
    self.networkClones[1]:evaluate()
  end
end
