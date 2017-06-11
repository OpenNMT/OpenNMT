--[[ Google's NMT encoder as described in https://arxiv.org/abs/1609.08144.

Bidirectional RNN on the first layer and unidirectional for the rest with residual
connections.

]]
local GoogleEncoder, parent = torch.class('onmt.GoogleEncoder', 'nn.Container')

function GoogleEncoder.declareOpts(cmd)
  onmt.Encoder.declareOpts(cmd)
end

function GoogleEncoder:__init(args, input)
  parent.__init(self)

  assert(args.layers > 1, 'GoogleEncoder only supports 2 layers or more')

  -- First layer is a BiEncoder.
  local brnnArgs = onmt.utils.Tensor.deepClone(args)
  brnnArgs.layers = 1
  brnnArgs.brnn_merge = 'concat'

  self:add(onmt.BiEncoder(brnnArgs, input))

  -- Remaining layers as a standard Encoder.
  local rnnArgs = onmt.utils.Tensor.deepClone(args)
  rnnArgs.layers = args.layers - 1
  rnnArgs.residual = true
  if args.dropout > 0 then
    rnnArgs.dropout_input = true
  end

  input = nn.Identity()
  input.inputSize = brnnArgs.rnn_size

  self:add(onmt.Encoder(rnnArgs, input))

  self.args = {}
  self.args.hiddenSize = args.rnn_size
  self.args.numEffectiveLayers = self.modules[1].args.numEffectiveLayers + self.modules[2].args.numEffectiveLayers

  self:resetPreallocation()
end

--[[ Return a new GoogleEncoder using the serialized data `pretrained`. ]]
function GoogleEncoder.load(pretrained)
  local self = torch.factory('onmt.GoogleEncoder')()
  parent.__init(self)

  self:add(onmt.BiEncoder.load(pretrained.modules[1]))
  self:add(onmt.Encoder.load(pretrained.modules[2]))
  self.args = pretrained.args

  self:resetPreallocation()
  return self
end

--[[ Return data to serialize. ]]
function GoogleEncoder:serialize()
  local modulesData = {}
  for i = 1, #self.modules do
    table.insert(modulesData, self.modules[i]:serialize())
  end

  return {
    name = 'GoogleEncoder',
    modules = modulesData,
    args = self.args
  }
end

function GoogleEncoder:resetPreallocation()
  self.gradContextProto = torch.Tensor()
end

function GoogleEncoder:forward(batch)
  local firstStates, firstContext = self.modules[1]:forward(batch)
  local nextInput = onmt.data.BatchTensor.new(firstContext, batch.sourceSize)
  local remainingStates, lastContext = self.modules[2]:forward(nextInput)

  local states = {}
  onmt.utils.Table.append(states, firstStates)
  onmt.utils.Table.append(states, remainingStates)

  if self.train then
    self.inputs = { batch, nextInput }
  end

  return states, lastContext
end

function GoogleEncoder:backward(batch, gradStatesOutput, gradContextOutput)
  local brnnGradStates = onmt.utils.Table.subrange(gradStatesOutput,
                                                   1,
                                                   self.modules[1].args.numEffectiveLayers)
  local rnnGradStates = onmt.utils.Table.subrange(gradStatesOutput,
                                                  self.modules[1].args.numEffectiveLayers + 1,
                                                  self.modules[2].args.numEffectiveLayers)

  local firstGradInputs = self.modules[2]:backward(self.inputs[2], rnnGradStates, gradContextOutput)

  gradContextOutput = onmt.utils.Tensor.reuseTensor(
    self.gradContextProto,
    { batch.size, #firstGradInputs, self.args.hiddenSize })

  for t = 1, #firstGradInputs do
    gradContextOutput[{{}, t}]:copy(firstGradInputs[t])
  end

  local gradInputs = self.modules[1]:backward(self.inputs[1], brnnGradStates, gradContextOutput)

  return gradInputs
end
