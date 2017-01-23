--[[ CudnnEncoder is a unidirectional/bidirectional Sequencer used for the source language
  using cudnn RNN implementation

    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     .      .      .             .
     |      |      |             |
    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     |      |      |             |
    x_1    x_2    x_3           x_n


Inherits from [onmt.Sequencer](onmt+modules+Sequencer).
--]]
local CudnnEncoder, parent = torch.class('onmt.CudnnEncoder', 'onmt.Sequencer')

--[[ Construct an encoder layer.

Parameters:

  * `inputNetwork` - input module.
  * `rnn` - recurrent module.
]]
function CudnnEncoder:__init(layers, inputSize, rnnSize, dropout, brnn, inputNetwork)
  self.rnn = onmt.utils.Cuda.cudnn.LSTM(inputSize, rnnSize, layers, false, dropout, true)
  if brnn then
    self.rnn.bidirectional = 'CUDNN_BIDIRECTIONAL'
  end

  self.inputNet = inputNetwork

  self.args = {}
  self.args.inputSize = inputSize

  parent.__init(self, inputNetwork)

  self:resetPreallocation()
end

function CudnnEncoder:resetPreallocation()
  self.inputProto = torch.Tensor()
  self.hiddenOutputProto = torch.Tensor()
  self.cellOutputProto = torch.Tensor()
  self.gradHiddenOutputProto = torch.Tensor()
  self.gradCellOutputProto = torch.Tensor()
end

--[[Compute the context representation of an input.

Parameters:

  * `batch` - as defined in batch.lua.

Returns:

  1. - final hidden states
  2. - context matrix H
--]]
function CudnnEncoder:forward(batch)

  -- Preallocated output matrix.
  self.inputs = onmt.utils.Tensor.reuseTensor(self.inputProto,
                                                { batch.sourceLength, batch.size, self.args.inputSize })

  -- to keep final hidden output for decoder
  local states = {}

  -- Act like nn.Sequential and call each clone in a feed-forward fashion.
  -- Calculate input layer
  for t = 1, batch.sourceLength do
    -- Construct "inputs". Prev states come first then source.
    self.inputs[{t, {}}]:copy(self:net(t):forward(batch:getSourceInput(t)))
  end

  self.rnn.hiddenOutput = onmt.utils.Tensor.reuseTensor(self.hiddenOutputProto,
                                              { self.rnn.numLayers * self.rnn.numDirections, batch.size, self.rnn.hiddenSize })
  self.rnn.cellOutput = onmt.utils.Tensor.reuseTensor(self.cellOutputProto,
                                            { self.rnn.numLayers * self.rnn.numDirections, batch.size, self.rnn.hiddenSize })

  local context = self.rnn:forward(self.inputs)

  -- dimension of context is seq x batch x rnn - we need it to be batch x seq x rnn
  context = context:transpose(1,2)

  for i=1, self.rnn.numLayers * self.rnn.numDirections, self.rnn.numDirections do
    table.insert(states, self.rnn.hiddenOutput[i])
    table.insert(states, self.rnn.cellOutput[i])
  end

  return states, context
end

--[[ Backward pass (only called during training)

  Parameters:

  * `batch` - must be same as for forward
  * `gradStatesOutput` gradient of loss wrt last state - this can be null if states are not used
  * `gradContextOutput` - gradient of loss wrt full context.

  Returns: `gradInputs` of input network.
--]]
function CudnnEncoder:backward(batch, gradStatesOutput, gradContextOutput)

  self.rnn.gradHiddenOutput = onmt.utils.Tensor.reuseTensor(self.gradHiddenOutputProto,
                                                 { self.rnn.numLayers * self.rnn.numDirections, batch.size, self.rnn.hiddenSize })
  self.rnn.gradCellOutput = onmt.utils.Tensor.reuseTensor(self.cellOutputProto,
                                                { self.rnn.numLayers * self.rnn.numDirections, batch.size, self.rnn.hiddenSize })

  for i=1, self.rnn.numLayers do
    self.rnn.gradHiddenOutput[i]:copy(gradStatesOutput[2*i-1])
    self.rnn.gradCellOutput[i]:copy(gradStatesOutput[2*i])
  end

  local gradInputs = self.rnn:backward(self.inputs, gradContextOutput:transpose(1,2))

  for t = 1, batch.sourceLength do
    -- Construct "inputs". Prev states come first then source.
    self:net(t):backward(batch:getSourceInput(t), gradInputs[{t, {}}])
  end

  return {}
end
