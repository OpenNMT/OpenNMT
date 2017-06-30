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

  self.args.numStates = self.fwd.args.numStates

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
  self.args.numStates = self.args.numStates or self.args.numEffectiveLayers

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

function BiEncoder:forward(batch, initial_states)
  assert(not initial_states, "Cannot apply bidirectional Encoder incrementally")

  if self.statesProto == nil then
    self.statesProto = onmt.utils.Tensor.initTensorTable(self.args.numStates,
                                                         self.stateProto,
                                                         { batch.size, self.args.hiddenSize })
  end

  local states = onmt.utils.Tensor.reuseTensorTable(self.statesProto, { batch.size, self.args.hiddenSize })
  local context = onmt.utils.Tensor.reuseTensor(self.contextProto,
                                                { batch.size, batch.sourceLength, self.args.hiddenSize })

  local fwdStates, fwdContext = self.fwd:forward(batch)
  batch:reverseSourceInPlace()
  local bwdStates, bwdContext = self.bwd:forward(batch)
  batch:reverseSourceInPlace()

  -- Merge final states.
  for i = 1, #states do
    if self.args.brnn_merge == 'concat' then
      states[i]:narrow(2, 1, self.args.rnn_size):copy(fwdStates[i])
      states[i]:narrow(2, self.args.rnn_size + 1, self.args.rnn_size):copy(bwdStates[i])
    elseif self.args.brnn_merge == 'sum' then
      states[i]:copy(fwdStates[i])
      states[i]:add(bwdStates[i])
    end
  end

  -- Merge outputs.
  if batch:variableLengths() then
    for b = 1, batch.size do
      local window = {b, {batch.sourceLength - batch.sourceSize[b] + 1, batch.sourceLength}}
      local reversedIndices = torch.linspace(batch.sourceSize[b], 1, batch.sourceSize[b]):long()

      -- Reverse outputs of the reversed encoder to align them with the regular outputs.
      local bwdOutputs = bwdContext[window]:index(1, reversedIndices)
      local fwdOutputs = fwdContext[window]

      if self.args.brnn_merge == 'concat' then
        context[window]:narrow(2, 1, self.args.rnn_size):copy(fwdOutputs)
        context[window]:narrow(2, self.args.rnn_size + 1, self.args.rnn_size):copy(bwdOutputs)
      elseif self.args.brnn_merge == 'sum' then
        context[window]:copy(fwdOutputs)
        context[window]:add(bwdOutputs)
      end
    end
  else
    for t = 1, batch.sourceLength do
      if self.args.brnn_merge == 'concat' then
        context[{{}, t, {1, self.args.rnn_size}}]
          :copy(fwdContext[{{}, t}])
        context[{{}, t, {self.args.rnn_size + 1, self.args.rnn_size * 2}}]
          :copy(bwdContext[{{}, -t}])
      elseif self.args.brnn_merge == 'sum' then
        context[{{}, t}]:copy(fwdContext[{{}, t}])
        context[{{}, t}]:add(bwdContext[{{}, -t}])
      end
    end
  end

  return states, context
end

function BiEncoder:backward(batch, gradStatesOutput, gradContextOutput)
  gradStatesOutput = gradStatesOutput
    or onmt.utils.Tensor.initTensorTable(self.args.numStates,
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

  local gradContextBwd = onmt.utils.Tensor.reuseTensor(
    self.gradContextBwdProto,
    { batch.size, batch.sourceLength, self.args.rnn_size })

  -- Reverse output gradients.
  if batch:variableLengths() then
    for b = 1, batch.size do
      local window = {b, {batch.sourceLength - batch.sourceSize[b] + 1, batch.sourceLength}}
      local reversedIndices = torch.linspace(batch.sourceSize[b], 1, batch.sourceSize[b]):long()
      gradContextBwd[window]:copy(gradContextOutputBwd[window]:index(1, reversedIndices))
    end
  else
    for t = 1, batch.sourceLength do
      gradContextBwd[{{}, t}]:copy(gradContextOutputBwd[{{}, -t}])
    end
  end

  batch:reverseSourceInPlace()
  local gradInputBwd = self.bwd:backward(batch, gradStatesOutputBwd, gradContextBwd)
  batch:reverseSourceInPlace()

  -- Add gradients coming from both directions.
  if batch:variableLengths() then
    for b = 1, batch.size do
      local padSize = batch.sourceLength - batch.sourceSize[b]
      for t = padSize + 1, batch.sourceLength do
        onmt.utils.Tensor.recursiveAdd(gradInputFwd[t],
                                       gradInputBwd[batch.sourceLength - t + 1 + padSize], b)
      end
    end
  else
    for t = 1, batch.sourceLength do
      onmt.utils.Tensor.recursiveAdd(gradInputFwd[t], gradInputBwd[batch.sourceLength - t + 1])
    end
  end

  return gradInputFwd
end
