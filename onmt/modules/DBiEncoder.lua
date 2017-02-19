--[[ DBiEncoder is a deep bidirectional Sequencer used for the source language.


--]]
local DBiEncoder, parent = torch.class('onmt.DBiEncoder', 'nn.Container')

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
  parent.__init(self)

  self.args = onmt.utils.ExtendedCmdLine.getModuleOpts(args, options)
  self.args.layers = args.layers

  self.layers = {}

  args.layers = 1
  args.brnn_merge = 'sum'
  for _=1,self.args.layers do
    table.insert(self.layers, onmt.BiEncoder(args, input))
    local identity = nn.Identity()
    identity.inputSize = args.rnn_size
    input = identity
    self:add(self.layers[#self.layers])
  end
  args.layers = self.args.layers
  self.args.numEffectiveLayers = self.layers[1].args.numEffectiveLayers * self.args.layers
  self.args.hiddenSize = args.rnn_size

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

  -- Prototype for preallocated gradient context
  self.gradContextProto = torch.Tensor()
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
  self.inputs = { batch }
  self.lranges = {}
  for i = 1,#self.layers do
    local layerStates, layerContext = self.layers[i]:forward(self.inputs[i])
    if i ~= #self.layers then
      table.insert(self.inputs, onmt.data.BatchTensor.new(layerContext))
    else
      context:copy(layerContext)
    end
    table.insert(self.lranges, torch.LongTensor(stateIdx, #layerStates))
    for j = 1,#layerStates do
      states[stateIdx]:copy(layerStates[j])
      stateIdx = stateIdx + 1
    end
  end
  return states, context
end

function DBiEncoder:backward(batch, gradStatesOutput, gradContextOutput)
  for i = #self.layers, 1, -1 do
    local lrange_gradStatesOutput
    if gradStatesOutput then
      lrange_gradStatesOutput = gradStatesOutput:narrow(2, self.lranges[i])
    end
    local gradContextInput = self.layers[i]:backward(self.inputs[i], lrange_gradStatesOutput, gradContextOutput)
    if i ~= 1 then
      gradContextOutput = onmt.utils.Tensor.reuseTensor(self.gradContextProto,
                                              { batch.size, #gradContextInput, self.args.hiddenSize })
      for t = 1, #gradContextInput do
        gradContextOutput[{{},t,{}}]:copy(gradContextInput[t])
      end
    end
  end

  return gradContextOutput
end
