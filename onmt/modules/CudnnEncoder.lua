--[[ CudnnEncoder is a unidirectional/bidirectional Encoder used for the source language
  using CuDNN RNN implementation

    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     .      .      .             .
     |      |      |             |
    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     |      |      |             |
    x_1    x_2    x_3           x_n

--]]
local CudnnEncoder, parent = torch.class('onmt.CudnnEncoder', 'nn.Container')

--[[ Construct an encoder layer.

Parameters:

  * `inputNetwork` - input module
  * `rnnClass` - CuDNN recurrent module class
  * `layers` - number of layers
  * `inputSize` - input dimension
  * `rnnSize` - RNN hidden dimension
  * `dropout` - dropout value
]]
function CudnnEncoder:__init(inputNetwork, rnnClass, layers, inputSize, rnnSize, dropout)
  parent.__init(self)

  self.rnn = rnnClass(inputSize, rnnSize, layers, false, dropout, false)
  self.inputNetwork = inputNetwork

  self.net = nn.Sequential()
    :add(self.inputNetwork)
    :add(nn.Transpose({1, 2})) -- By default, batch is second in CuDNN.
    :add(self.rnn)

  self:add(self.net)

  self:resetPreallocation()
end

--[[ Return a new CudnnEncoder using the serialized data `pretrained`. ]]
function CudnnEncoder.load(pretrained)
  local self = torch.factory('onmt.CudnnEncoder')()

  parent.__init(self)

  self.net = pretrained.modules[1]
  self.inputNetwork = self.net:get(1)
  self.rnn = self.net:get(3)

  self.rnn:resetDropoutDescriptor()
  self.rnn:resetRNNDescriptor()
  self.rnn:resetIODescriptors()

  self:add(self.net)

  self:resetPreallocation()

  return self
end

--[[ Return data to serialize. ]]
function CudnnEncoder:serialize()
  return {
    name = 'CudnnEncoder',
    modules = self.modules
  }
end

function CudnnEncoder:resetPreallocation()
  self.rnn.gradHiddenOutput = torch.Tensor()
  self.rnn.gradCellOutput = torch.Tensor()
end

--[[Compute the context representation of an input.

Parameters:

  * `batch` - as defined in batch.lua.

Returns:

  1. - final hidden states
  2. - context matrix H

--]]
function CudnnEncoder:forward(batch)
  self.inputs = batch:getSourceInput()

  local context = self.net:forward(self.inputs):transpose(1, 2)

  local states = {}

  for i = 1, self.rnn.numLayers do
    if self.rnn.mode == 'CUDNN_LSTM' then
      table.insert(states, self.rnn.cellOutput[i])
    end
    table.insert(states, self.rnn.hiddenOutput[i])
  end

  return states, context
end

--[[ Backward pass (only called during training)

Parameters:

  * `batch` - must be same as for forward
  * `gradStatesOutput` gradient of loss wrt last state - this can be null if states are not used
  * `gradContextOutput` - gradient of loss wrt full context.

Returns:

  * `gradInputs` of input network.

--]]
function CudnnEncoder:backward(batch, gradStatesOutput, gradContextOutput)
  self.rnn:resizeHidden(self.rnn.gradHiddenOutput)
  self.rnn:resizeHidden(self.rnn.gradCellOutput)

  -- gradStatesOutput is nil for instance when using in LanguageModel.
  if gradStatesOutput then
    for i = 1, self.rnn.numLayers do
      self.rnn.gradHiddenOutput[i]:copy(gradStatesOutput[2 * i - 1]:narrow(2, 1, self.rnn.hiddenSize))
      if not self.rnn.mode == 'CUDNN_GRU' then
        self.rnn.gradCellOutput[i]:copy(gradStatesOutput[2 * i]:narrow(2, 1, self.rnn.hiddenSize))
      end
    end
  end

  return self.net:backward(self.inputs or batch:getSourceInput(), gradContextOutput:transpose(1, 2))
end
