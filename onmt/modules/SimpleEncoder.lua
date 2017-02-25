--[[ SimpleEncoder is a unidirectional Sequencer used for the source language.

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
local SimpleEncoder, parent = torch.class('onmt.SimpleEncoder', 'onmt.Sequencer')

local options = {
  {'-layers', 2,         [[Number of layers in the RNN SimpleEncoder/decoder]],
                            {valid=onmt.utils.ExtendedCmdLine.isUInt()}},
  {'-rnn_size', 500,     [[Size of RNN hidden states]],
                            {valid=onmt.utils.ExtendedCmdLine.isUInt()}},
  {'-rnn_type', 'LSTM', [[Type of RNN cell]],
                            {enum={'LSTM','GRU'}}},
  {'-dropout', 0.3, [[Dropout probability. Dropout is applied between vertical LSTM stacks.]]},
  {'-residual', false, [[Add residual connections between RNN layers.]]}
}

function SimpleEncoder.declareOpts(cmd)
  cmd:setCmdLineOptions(options)
end

--[[ Construct an SimpleEncoder layer.

Parameters:

  * `inputNetwork` - input module.
  * `rnn` - recurrent module.
]]
function SimpleEncoder:__init(args, inputNetwork)
  local RNN = onmt.LSTM
  if args.rnn_type == 'GRU' then
    RNN = onmt.GRU
  end

  local rnn = RNN.new(args.layers, inputNetwork.inputSize, args.rnn_size, args.dropout, args.residual)

  self.rnn = rnn
  self.inputNet = inputNetwork

  self.args = {}
  self.args.rnnSize = self.rnn.outputSize
  self.args.numEffectiveLayers = self.rnn.numEffectiveLayers

  parent.__init(self, self:_buildModel())

  self:resetPreallocation()
end

--[[ Return a new SimpleEncoder using the serialized data `pretrained`. ]]
function SimpleEncoder.load(pretrained)
  local self = torch.factory('onmt.SimpleEncoder')()

  self.args = pretrained.args
  parent.__init(self, pretrained.modules[1])

  self:resetPreallocation()

  return self
end

--[[ Return data to serialize. ]]
function SimpleEncoder:serialize()
  return {
    name = 'SimpleEncoder',
    modules = self.modules,
    args = self.args
  }
end

function SimpleEncoder:resetPreallocation()
  -- Prototype for preallocated hidden and cell states.
  self.stateProto = torch.Tensor()

  -- Prototype for preallocated output gradients.
  self.gradOutputProto = torch.Tensor()

  -- Prototype for preallocated context vector.
  self.contextProto = torch.Tensor()
end

function SimpleEncoder:maskPadding()
  self.maskPad = true
end

--[[ Build one time-step of an SimpleEncoder

Returns: An nn-graph mapping

  $${(c^1_{t-1}, h^1_{t-1}, .., c^L_{t-1}, h^L_{t-1}, x_t) =>
  (c^1_{t}, h^1_{t}, .., c^L_{t}, h^L_{t})}$$

  Where $$c^l$$ and $$h^l$$ are the hidden and cell states at each layer,
  $$x_t$$ is a sparse word to lookup.
--]]
function SimpleEncoder:_buildModel()
  local inputs = {}
  local states = {}

  -- Inputs are previous layers first.
  for _ = 1, self.args.numEffectiveLayers do
    local h0 = nn.Identity()() -- batchSize x rnnSize
    table.insert(inputs, h0)
    table.insert(states, h0)
  end

  -- Input word.
  local x = nn.Identity()() -- batchSize
  table.insert(inputs, x)

  -- Compute input network.
  local input = self.inputNet(x)
  table.insert(states, input)

  -- Forward states and input into the RNN.
  local outputs = self.rnn(states)
  return nn.gModule(inputs, { outputs })
end

--[[Compute the context representation of an input.

Parameters:

  * `batch` - as defined in batch.lua.

Returns:

  1. - final hidden states
  2. - context matrix H
--]]
function SimpleEncoder:forward(batch)

  -- TODO: Change `batch` to `input`.

  local finalStates
  local outputSize = self.args.rnnSize

  if self.statesProto == nil then
    self.statesProto = onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                                         self.stateProto,
                                                         { batch.size, outputSize })
  end

  -- Make initial states h_0.
  local states = onmt.utils.Tensor.reuseTensorTable(self.statesProto, { batch.size, outputSize })

  -- Preallocated output matrix.
  local context = onmt.utils.Tensor.reuseTensor(self.contextProto,
                                                { batch.size, batch.sourceLength, outputSize })

  if self.maskPad and not batch.sourceInputPadLeft then
    finalStates = onmt.utils.Tensor.recursiveClone(states)
  end
  if self.train then
    self.inputs = {}
  end

  -- Act like nn.Sequential and call each clone in a feed-forward
  -- fashion.
  for t = 1, batch.sourceLength do

    -- Construct "inputs". Prev states come first then source.
    local inputs = {}
    onmt.utils.Table.append(inputs, states)
    table.insert(inputs, batch:getSourceInput(t))

    if self.train then
      -- Remember inputs for the backward pass.
      self.inputs[t] = inputs
    end
    states = self:net(t):forward(inputs)

    -- Make sure it always returns table.
    if type(states) ~= "table" then states = { states } end

    -- Special case padding.
    if self.maskPad then
      for b = 1, batch.size do
        if batch.sourceInputPadLeft and t <= batch.sourceLength - batch.sourceSize[b] then
          for j = 1, #states do
            states[j][b]:zero()
          end
        elseif not batch.sourceInputPadLeft and t == batch.sourceSize[b] then
          for j = 1, #states do
            finalStates[j][b]:copy(states[j][b])
          end
        end
      end
    end

    -- Copy output (h^L_t = states[#states]) to context.
    context[{{}, t}]:copy(states[#states])
  end

  if finalStates == nil then
    finalStates = states
  end

  return finalStates, context
end

--[[ Backward pass (only called during training)

  Parameters:

  * `batch` - must be same as for forward
  * `gradStatesOutput` gradient of loss wrt last state - this can be null if states are not used
  * `gradContextOutput` - gradient of loss wrt full context.

  Returns: `gradInputs` of input network.
--]]
function SimpleEncoder:backward(batch, gradStatesOutput, gradContextOutput)
  -- TODO: change this to (input, gradOutput) as in nngraph.
  local outputSize = self.args.rnnSize
  if self.gradOutputsProto == nil then
    self.gradOutputsProto = onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                                              self.gradOutputProto,
                                                              { batch.size, outputSize })
  end

  local gradStatesInput
  if gradStatesOutput then
    gradStatesInput = onmt.utils.Tensor.copyTensorTable(self.gradOutputsProto, gradStatesOutput)
  else
    -- if gradStatesOutput is not defined - start with empty tensor
    gradStatesInput = onmt.utils.Tensor.reuseTensorTable(self.gradOutputsProto, { batch.size, outputSize })
  end

  local gradInputs = {}

  for t = batch.sourceLength, 1, -1 do
    -- Add context gradients to last hidden states gradients.
    gradStatesInput[#gradStatesInput]:add(gradContextOutput[{{}, t}])

    local gradInput = self:net(t):backward(self.inputs[t], gradStatesInput)

    -- Prepare next SimpleEncoder output gradients.
    for i = 1, #gradStatesInput do
      gradStatesInput[i]:copy(gradInput[i])
    end

    -- Gather gradients of all user inputs.
    gradInputs[t] = {}
    for i = #gradStatesInput + 1, #gradInput do
      table.insert(gradInputs[t], gradInput[i])
    end

    if #gradInputs[t] == 1 then
      gradInputs[t] = gradInputs[t][1]
    end
  end
  -- TODO: make these names clearer.
  -- Useful if input came from another network.
  return gradInputs

end
