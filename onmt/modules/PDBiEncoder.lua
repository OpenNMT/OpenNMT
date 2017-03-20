--[[ PDBiEncoder is a pyramidal deep bidirectional Sequencer used for the source language.


--]]
local PDBiEncoder, parent = torch.class('onmt.PDBiEncoder', 'nn.Container')

local options = {
  {'-pdbrnn_reduction', 2, [[Time-Reduction factor at each layer.]]}
}

function PDBiEncoder.declareOpts(cmd)
  onmt.BiEncoder.declareOpts(cmd)
  cmd:setCmdLineOptions(options)
end


--[[ Create a pyrimal deep bidirectional encoder - each layers reconnect before starting another bidirectional layer

Parameters:

  * `args` - global arguments
  * `input` - input neural network.
]]
function PDBiEncoder:__init(args, input)
  parent.__init(self)

  self.args = onmt.utils.ExtendedCmdLine.getModuleOpts(args, options)
  self.args.layers = args.layers
  self.args.dropout = args.dropout
  local dropout_input = args.dropout_input

  self.layers = {}

  args.layers = 1
  args.brnn_merge = 'sum'
  self.args.multiplier = 1
  for _ = 1,self.args.layers do
    table.insert(self.layers, onmt.BiEncoder(args, input))
    local identity = nn.Identity()
    identity.inputSize = args.rnn_size
    input = identity
    self:add(self.layers[#self.layers])
    if #self.layers ~= 1 then
      self.args.multiplier = self.args.multiplier * self.args.pdbrnn_reduction
    else
      -- trick to force a dropout on each layer L > 1
      if args.dropout > 0 then
        args.dropout_input = true
      end
    end
  end
  args.layers = self.args.layers
  args.dropout_input = dropout_input
  self.args.numEffectiveLayers = self.layers[1].args.numEffectiveLayers * self.args.layers
  self.args.hiddenSize = args.rnn_size

  self:resetPreallocation()
end

--[[ Return a new PDBiEncoder using the serialized data `pretrained`. ]]
function PDBiEncoder.load(pretrained)
  local self = torch.factory('onmt.PDBiEncoder')()
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
function PDBiEncoder:serialize()
  local layersData = {}
  for i = 1, #self.layers do
    table.insert(layersData, self.layers[i]:serialize())
  end

  return {
    name = 'PDBiEncoder',
    layers = layersData,
    args = self.args
  }
end

function PDBiEncoder:resetPreallocation()
  -- Prototype for preallocated full context vector.
  self.contextProto = torch.Tensor()

  -- Prototype for preallocated full hidden states tensors.
  self.stateProto = torch.Tensor()

  -- Prototype for preallocated gradient context
  self.gradContextProto = torch.Tensor()
end

function PDBiEncoder:maskPadding()
  self.layers[1]:maskPadding()
end

-- size of context vector
function PDBiEncoder:contextSize(sourceSize, sourceLength)
  local contextLength = math.ceil(batch_length/self.args.multiplier)
  local contextSize = {}
  for i = 1, #sourceSize do
    table.insert(contextSize, math.ceil(contextSize[i]/self.args.multiplier))
  end
  return sourceSize, sourceLength
end

function PDBiEncoder:forward(batch)
  -- adjust batch length so that it can be divided
  local batch_length = batch.sourceLength
  batch_length = math.ceil(batch_length/self.args.multiplier)*self.args.multiplier
  batch.sourceLength = batch_length

  if self.statesProto == nil then
    self.statesProto = onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                                         self.stateProto,
                                                         { batch.size, self.args.hiddenSize })
  end

  local states = onmt.utils.Tensor.reuseTensorTable(self.statesProto, { batch.size, self.args.hiddenSize })
  local context

  local stateIdx = 1
  self.inputs = { batch }
  self.lranges = {}
  for i = 1,#self.layers do
    local layerStates, layerContext = self.layers[i]:forward(self.inputs[i])
    if i ~= #self.layers then
      -- compress the layer Context along time dimension
      local storageOffset = layerContext:storageOffset()
      local strideReduced = layerContext:stride()
      strideReduced[2] = strideReduced[2] * self.args.pdbrnn_reduction
      local sizeReduced = layerContext:size()
      sizeReduced[2] = math.floor(sizeReduced[2] / self.args.pdbrnn_reduction)
      local reducedContext = layerContext
      reducedContext:set(layerContext:storage(), storageOffset, sizeReduced, strideReduced)
      for j = 1, self.args.pdbrnn_reduction-1 do
        local to_add = layerContext
        to_add:set(layerContext:storage(), storageOffset+j, sizeReduced, strideReduced)
        reducedContext:add(to_add)
      end
      table.insert(self.inputs, onmt.data.BatchTensor.new(reducedContext))
      -- record what is the size of the last reduction
      batch.encoderOutputLength = sizeReduced[2]
    else
      context = onmt.utils.Tensor.reuseTensor(self.contextProto, layerContext:size())
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

function PDBiEncoder:backward(batch, gradStatesOutput, gradContextOutput)
  for i = #self.layers, 1, -1 do
    local lrange_gradStatesOutput
    if gradStatesOutput then
      lrange_gradStatesOutput = onmt.utils.Table.subrange(gradStatesOutput, self.lranges[i][1], self.lranges[i][2])
    end
    local gradContextInput = self.layers[i]:backward(self.inputs[i], lrange_gradStatesOutput, gradContextOutput)
    if i ~= 1 then
      gradContextOutput = onmt.utils.Tensor.reuseTensor(self.gradContextProto,
                                              { batch.size, #gradContextInput*self.args.pdbrnn_reduction, self.args.hiddenSize })
      for t = 1, #gradContextInput do
        for j = 1, self.args.pdbrnn_reduction do
          gradContextOutput[{{},self.args.pdbrnn_reduction*(t-1)+j,{}}]:copy(gradContextInput[t])
        end
      end
    end
  end

  return gradContextOutput
end
