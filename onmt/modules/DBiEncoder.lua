--[[ DBiEncoder is a bidirectional Sequencer used for the source language.


--]]
local DBiEncoder, parent = torch.class('onmt.DBiEncoder', 'onmt.ComplexEncoder')

local options = {}

function DBiEncoder.declareOpts(cmd)
  onmt.BiEncoder.declareOpts(cmd)
  cmd:setCmdLineOptions(options)
end


--[[ Create a deep bidirectional encoder - each layers reconnect

Parameters:

  * `args` - global arguments
  * `input` - input neural network.
]]
function DBiEncoder:__init(args, input)
  self.args = onmt.utils.ExtendedCmdLine.getModuleOpts(args, options)
  self.args.layers = args.layers

  self.layers = {}

  args.layers = 1
  args.brnn_merge = 'sum'
  for _=1,self.args.layers do
    table.insert(self.layers, onmt.BiEncoder(args, input))
    input = nn.Identity()
  end
  args.layers = self.args.layers
  self.args.numEffectiveLayers = self.layers[1].args.numEffectiveLayers * self.args.layers

  parent.__init(self, self.layers)

  self:resetPreallocation()
end

--[[ Return a new DBiEncoder using the serialized data `pretrained`. ]]
function DBiEncoder.load(pretrained)
  local self = torch.factory('onmt.DBiEncoder')()

  for i=1, #pretrained.modules do
    self.layers = onmt.Encoder.load(pretrained.modules[i])
  end

  self.args = pretrained.args

  parent.__init(self, self.layers)

  self:resetPreallocation()

  return self
end

--[[ Return data to serialize. ]]
function DBiEncoder:serialize()
  local modulesData = {}
  for i = 1, #self.modules do
    table.insert(modulesData, self.modules[i]:serialize())
  end

  return {
    name = 'DBiEncoder',
    modules = modulesData,
    args = self.args
  }
end

function DBiEncoder:resetPreallocation()
  -- Prototype for preallocated full context vector.
  self.contextProto = torch.Tensor()

  -- Prototype for preallocated full hidden states tensors.
  self.stateProto = torch.Tensor()

  -- Prototype for preallocated gradient of the backward context
  self.gradContextBwdProto = torch.Tensor()
end

function DBiEncoder:maskPadding()
  self.layers[1]:maskPadding()
end

function DBiEncoder:forward(batch)
  if self.statesProto == nil then
    self.statesProto = onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                                         self.stateProto,
                                                         { batch.size, self.args.hiddenSize })
  end

  local states = onmt.utils.Tensor.reuseTensorTable(self.statesProto, { batch.size, self.args.hiddenSize })
  local context = onmt.utils.Tensor.reuseTensor(self.contextProto,
                                                { batch.size, batch.sourceLength, self.args.hiddenSize })

  local stateIdx = 1
  for i = 1,#self.layers do
    local layerStates, layerContext = self.layers[i]:forward(batch)
    batch = layerContext
    for j = 1,#layerStates do
      states[stateIdx]:copy(layerStates[j])
      stateIdx = stateIdx + 1
    end
    if i == #self.layers then
      context:copy(layerContext)
    end
  end

  return states, context
end

function DBiEncoder:backward(batch, gradStatesOutput, gradContextOutput)
  gradStatesOutput = gradStatesOutput
    or onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                         onmt.utils.Cuda.convert(torch.Tensor()),
                                         { batch.size, self.args.hiddenSize })

  local gradContextOutputFwd
  local gradContextOutputBwd

  local gradStatesOutputFwd = {}
  local gradStatesOutputBwd = {}

  if self.args.brnn_merge == 'concat' then
    local gradContextOutputSplit = gradContextOutput:chunk(2, 3)
    gradContextOutputFwd = gradContextOutputSplit[1]
    gradContextOutputBwd = gradContextOutputSplit[2]

    for i = 1, #gradStatesOutput do
      local statesSplit = gradStatesOutput[i]:chunk(2, 2)
      table.insert(gradStatesOutputFwd, statesSplit[1])
      table.insert(gradStatesOutputBwd, statesSplit[2])
    end
  elseif self.args.brnn_merge == 'sum' then
    gradContextOutputFwd = gradContextOutput
    gradContextOutputBwd = gradContextOutput

    gradStatesOutputFwd = gradStatesOutput
    gradStatesOutputBwd = gradStatesOutput
  end

  local gradInputFwd = self.fwd:backward(batch, gradStatesOutputFwd, gradContextOutputFwd)

  -- reverse gradients of the backward context
  local gradContextBwd = onmt.utils.Tensor.reuseTensor(self.gradContextBwdProto,
                                                       { batch.size, batch.sourceLength, self.args.rnn_size })

  for t = 1, batch.sourceLength do
    gradContextBwd[{{}, t}]:copy(gradContextOutputBwd[{{}, batch.sourceLength - t + 1}])
  end

  local gradInputBwd = self.bwd:backward(batch, gradStatesOutputBwd, gradContextBwd)

  for t = 1, batch.sourceLength do
    onmt.utils.Tensor.recursiveAdd(gradInputFwd[t], gradInputBwd[batch.sourceLength - t + 1])
  end

  return gradInputFwd
end
