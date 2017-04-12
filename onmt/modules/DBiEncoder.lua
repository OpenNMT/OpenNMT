--[[ DBiEncoder is a deep bidirectional Sequencer used for the source language.


--]]
local DBiEncoder, parent = torch.class('onmt.DBiEncoder', 'nn.Container')

local options = {}

function DBiEncoder.declareOpts(cmd)
  onmt.BiEncoder.declareOpts(cmd)
  cmd:setCmdLineOptions(options)
end


--[[ Create a deep bidirectional encoder - each layers reconnect before starting another bidirectional layer

Parameters:

  * `args` - global arguments
  * `input` - input neural network.
]]
function DBiEncoder:__init(args, input)
  parent.__init(self)

  self.args = onmt.utils.ExtendedCmdLine.getModuleOpts(args, options)
  self.args.layers = args.layers
  self.args.dropout = args.dropout
  local dropout_input = args.dropout_input

  self.layers = {}

  args.layers = 1
  args.brnn_merge = 'sum'

  for _= 1, self.args.layers do
    table.insert(self.layers, onmt.BiEncoder(args, input))
    local identity = nn.Identity()
    identity.inputSize = args.rnn_size
    input = identity
    self:add(self.layers[#self.layers])
    -- trick to force a dropout on each layer L > 1
    if #self.layers == 1 and args.dropout > 0 then
      args.dropout_input = true
    end
  end
  args.layers = self.args.layers
  self.args.numEffectiveLayers = self.layers[1].args.numEffectiveLayers * self.args.layers
  self.args.hiddenSize = args.rnn_size
  args.dropout_input = dropout_input

  self:resetPreallocation()
end

--[[ Return a new DBiEncoder using the serialized data `pretrained`. ]]
function DBiEncoder.load(pretrained)
  local self = torch.factory('onmt.DBiEncoder')()
  parent.__init(self)

  self.layers = {}

  for i=1, #pretrained.layers do
    self.layers[i] = onmt.BiEncoder.load(pretrained.layers[i])
  end

  self.args = pretrained.args

  self:resetPreallocation()
  return self
end

--[[ Return data to serialize. ]]
function DBiEncoder:serialize()
  local layersData = {}

  for i = 1, #self.layers do
    table.insert(layersData, self.layers[i]:serialize())
  end

  return {
    name = 'DBiEncoder',
    layers = layersData,
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
    table.insert(self.lranges, {stateIdx, #layerStates})
    for j = 1,#layerStates do
      states[stateIdx]:copy(layerStates[j])
      stateIdx = stateIdx + 1
    end
  end
  return states, context
end

function DBiEncoder:backward(batch, gradStatesOutput, gradContextOutput)
  local gradInputs

  for i = #self.layers, 1, -1 do
    local lrange_gradStatesOutput
    if gradStatesOutput then
      lrange_gradStatesOutput = gradStatesOutput[{}]
    end
    gradInputs = self.layers[i]:backward(self.inputs[i], lrange_gradStatesOutput, gradContextOutput)
    if i ~= 1 then
      gradContextOutput = onmt.utils.Tensor.reuseTensor(self.gradContextProto,
                                              { batch.size, #gradInputs, self.args.hiddenSize })
      for t = 1, #gradInputs do
        gradContextOutput[{{},t,{}}]:copy(gradInputs[t])
      end
    end
  end

  return gradInputs
end
