---------------------------------------------------------------------------------
-- Local utility functions
---------------------------------------------------------------------------------

local function reverseInput(batch)
  batch.sourceInput, batch.sourceInputRev = batch.sourceInputRev, batch.sourceInput
  batch.sourceInputFeatures, batch.sourceInputRevFeatures = batch.sourceInputRevFeatures, batch.sourceInputFeatures
  batch.sourceInputPadLeft, batch.sourceInputRevPadLeft = batch.sourceInputRevPadLeft, batch.sourceInputPadLeft
end

---------------------------------------------------------------------------------

--[[ BiEncoder is a bidirectional Sequencer used for the source language.


 `netFwd`

    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     .      .      .             .
     |      |      |             |
    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     |      |      |             |
    x_1    x_2    x_3           x_n

 `netBwd`

    h_1 <= h_2 <= h_3 <= ... <= h_n
     |      |      |             |
     .      .      .             .
     |      |      |             |
    h_1 <= h_2 <= h_3 <= ... <= h_n
     |      |      |             |
     |      |      |             |
    x_1    x_2    x_3           x_n

Inherits from [onmt.Sequencer](onmt+modules+Sequencer).

--]]
local BiEncoder, parent = torch.class('onmt.BiEncoder', 'nn.Container')

local options = {
  {
    '-brnn_merge', 'sum',
    [[Merge action for the bidirectional states.]],
    {
      enum = {'concat', 'sum'},
      structural = 0
    }
  }
}

function BiEncoder.declareOpts(cmd)
  onmt.Encoder.declareOpts(cmd)
  cmd:setCmdLineOptions(options)
end


--[[ Create a bi-encoder.

Parameters:

  * `input` - input neural network.
  * `rnn` - recurrent template module.
  * `merge` - fwd/bwd merge operation {"concat", "sum"}
]]
function BiEncoder:__init(args, input)
  parent.__init(self)

  self.args = onmt.utils.ExtendedCmdLine.getModuleOpts(args, options)

  local orgRNNSize = args.rnn_size

  -- Compute rnn hidden size depending on hidden states merge action.
  if self.args.brnn_merge == 'concat' then
    if args.rnn_size % 2 ~= 0 then
      error('in concat mode, rnn_size must be divisible by 2')
    end
    args.rnn_size = args.rnn_size / 2
  end

  self.args.rnn_size = args.rnn_size

  self.fwd = onmt.Encoder.new(args, input)
  self.bwd = onmt.Encoder.new(args, input:clone('weight', 'bias', 'gradWeight', 'gradBias'))

  self.args.numEffectiveLayers = self.fwd.args.numEffectiveLayers

  if self.args.brnn_merge == 'concat' then
    self.args.hiddenSize = self.args.rnn_size * 2
  else
    self.args.hiddenSize = self.args.rnn_size
  end

  self:add(self.fwd)
  self:add(self.bwd)

  args.rnn_size = orgRNNSize

  self:resetPreallocation()
end

--[[ Return a new BiEncoder using the serialized data `pretrained`. ]]
function BiEncoder.load(pretrained)
  local self = torch.factory('onmt.BiEncoder')()

  parent.__init(self)

  self.fwd = onmt.Encoder.load(pretrained.modules[1])
  self.bwd = onmt.Encoder.load(pretrained.modules[2])
  self.args = pretrained.args

  -- backward compatibility
  self.args.rnn_size = self.args.rnn_size or self.args.rnnSize
  self.args.brnn_merge = self.args.brnn_merge or self.args.merge

  self:add(self.fwd)
  self:add(self.bwd)

  self:resetPreallocation()

  return self
end

--[[ Return data to serialize. ]]
function BiEncoder:serialize()
  local modulesData = {}
  for i = 1, #self.modules do
    table.insert(modulesData, self.modules[i]:serialize())
  end

  return {
    name = 'BiEncoder',
    modules = modulesData,
    args = self.args
  }
end

function BiEncoder:resetPreallocation()
  -- Prototype for preallocated full context vector.
  self.contextProto = torch.Tensor()

  -- Prototype for preallocated full hidden states tensors.
  self.stateProto = torch.Tensor()

  -- Prototype for preallocated gradient of the backward context
  self.gradContextBwdProto = torch.Tensor()
end

function BiEncoder:forward(batch)
  if self.statesProto == nil then
    self.statesProto = onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                                         self.stateProto,
                                                         { batch.size, self.args.hiddenSize })
  end

  local states = onmt.utils.Tensor.reuseTensorTable(self.statesProto, { batch.size, self.args.hiddenSize })
  local context = onmt.utils.Tensor.reuseTensor(self.contextProto,
                                                { batch.size, batch.sourceLength, self.args.hiddenSize })

  local fwdStates, fwdContext = self.fwd:forward(batch)
  reverseInput(batch)
  local bwdStates, bwdContext = self.bwd:forward(batch)
  reverseInput(batch)

  if self.args.brnn_merge == 'concat' then
    for i = 1, #fwdStates do
      states[i]:narrow(2, 1, self.args.rnn_size):copy(fwdStates[i])
      states[i]:narrow(2, self.args.rnn_size + 1, self.args.rnn_size):copy(bwdStates[i])
    end
    for t = 1, batch.sourceLength do
      context[{{}, t}]:narrow(2, 1, self.args.rnn_size)
        :copy(fwdContext[{{}, t}])
      context[{{}, t}]:narrow(2, self.args.rnn_size + 1, self.args.rnn_size)
        :copy(bwdContext[{{}, batch.sourceLength - t + 1}])
    end
  elseif self.args.brnn_merge == 'sum' then
    for i = 1, #states do
      states[i]:copy(fwdStates[i])
      states[i]:add(bwdStates[i])
    end
    for t = 1, batch.sourceLength do
      context[{{}, t}]:copy(fwdContext[{{}, t}])
      context[{{}, t}]:add(bwdContext[{{}, batch.sourceLength - t + 1}])
    end
  end

  return states, context
end

function BiEncoder:backward(batch, gradStatesOutput, gradContextOutput)
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
