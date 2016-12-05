local Data = require('lib.data')

--[[ Unit to decode a sequence of output tokens.

     .      .      .             .
     |      |      |             |
    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     .      .      .             .
     |      |      |             |
    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     |      |      |             |
    x_1    x_2    x_3           x_n

Inherits from [onmt.Sequencer](lib+onmt+Sequencer).

--]]
-- local Data = require "../data"
local Decoder, parent = torch.class('onmt.Decoder', 'onmt.Sequencer')


--[[ Construct an encoder layer.

Parameters:

  * `args` - global options.
  * `network` - optional, recurrent step template.
  * `generator` - optional, a output [onmt.Generator](lib+onmt+Generator).
--]]
function Decoder:__init(input_network, rnn, generator,
                        input_feed, mask_padding,
                        network)
  self.rnn = rnn
  self.inputNet = input_network
  self._rnn_size = self.rnn.output_size
  self._num_effective_layers = self.rnn.num_effective_layers
  self._input_feed = input_feed

  parent.__init(self, {}, network or self:_buildModel())

  -- The generator use the output of the decoder sequencer to generate the
  -- likelihoods over the target vocabulary.
  self.generator = generator
  self:add(self.generator)

  -- Input feeding means the decoder takes an extra
  -- vector each time representing the attention at the
  -- previous step.
  if input_feed then
    self.inputFeedProto = torch.Tensor()
  end

  -- Mask padding means that the attention-layer is constrained to
  -- give zero-weight to padding. This is done by storing a reference
  -- to the softmax attention-layer.
  if mask_padding then
    self.network:apply(function (layer)
      if layer.name == 'decoder_attn' then
        self.decoder_attn = layer
      end
    end)
  end

  -- Prototype for preallocated context gradient.
  self.gradContextProto = torch.Tensor()
end

--[[ Build a default one time-step of the decoder

Parameters:

  * `args` - global options.

Returns: An nn-graph mapping

  $${(c^1_{t-1}, h^1_{t-1}, .., c^L_{t-1}, h^L_{t-1}, x_t, con/H, if) =>
  (c^1_{t}, h^1_{t}, .., c^L_{t}, h^L_{t}, a)}$$

  Where ${c^l}$ and ${h^l}$ are the hidden and cell states at each layer,
  ${x_t}$ is a sparse word to lookup,
  ${con/H}$ is the context/source hidden states for attention,
  ${if}$ is the input feeding, and
  ${a}$ is the context vector computed at this timestep.
--]]
function Decoder:_buildModel()
  local inputs = {}
  local states = {}

  -- Inputs are previous layers first.
  for _ = 1, self._num_effective_layers do
    local h0 = nn.Identity()() -- batch_size x rnn_size
    table.insert(inputs, h0)
    table.insert(states, h0)
  end

  local x = nn.Identity()() -- batch_size
  table.insert(inputs, x)

  local context = nn.Identity()() -- batch_size x source_length x rnn_size
  table.insert(inputs, context)

  local input_feed
  if self._input_feed then
    input_feed = nn.Identity()() -- batch_size x rnn_size
    table.insert(inputs, input_feed)
  end

  -- Compute the input network.
  local input = self.inputNet(x)

  -- If set, concatenate previous decoder output.
  if self._input_feed then
    input = nn.JoinTable(2)({input, input_feed})
  end
  table.insert(states, input)

  -- Forward states and input into the RNN.
  local outputs = self.rnn(states)

  -- The output of a subgraph is a node: split it to access the last RNN output.
  outputs = { outputs:split(self._num_effective_layers) }

  -- Compute the attention here using h^L as query.
  local attn_layer = onmt.GlobalAttention(self._rnn_size)
  attn_layer.name = 'decoder_attn'
  local attn_output = attn_layer({outputs[#outputs], context})
  if self.rnn.dropout > 0 then
    attn_output = nn.Dropout(self.rnn.dropout)(attn_output)
  end
  table.insert(outputs, attn_output)
  return nn.gModule(inputs, outputs)
end

--[[ Update internals of model to prepare for new batch.

  Parameters:

  * See  [onmt.MaskedSoftmax](lib+onmt+MaskedSoftmax).
--]]
function Decoder:reset(source_sizes, source_length, beam_size)

  self.decoder_attn:replace(function(module)
    if module.name == 'softmax_attn' then
      local mod
      if source_sizes ~= nil then
        mod = onmt.MaskedSoftmax(source_sizes, source_length, beam_size)
      else
        mod = nn.SoftMax()
      end

      mod.name = 'softmax_attn'
      mod = utils.Cuda.convert(mod)
      self.softmax_attn = mod
      return mod
    else
      return module
    end
  end)
end

--[[ Run one step of the decoder.

Parameters:

 * `input` - sparse input (1)
 * `prev_states` - stack of hidden states (batch x layers*model x rnn_size)
 * `context` - encoder output (batch x n x rnn_size)
 * `prev_out` - previous distribution (batch x #words)
 * `t` - current timestep

Returns:

 1. `out` - Top-layer Hidden state
 2. `states` - All states
--]]
function Decoder:forward_one(input, prev_states, context, prev_out, t)
  local inputs = {}

  -- Create RNN input (see sequencer.lua `build_network('dec')`).
  utils.Table.append(inputs, prev_states)
  table.insert(inputs, input)
  table.insert(inputs, context)
  local input_size
  if torch.type(input) == 'table' then
    input_size = input[1]:size(1)
  else
    input_size = input:size(1)
  end

  if self._input_feed then
    if prev_out == nil then
      table.insert(inputs, utils.Tensor.reuseTensor(self.inputFeedProto,
                                                    { input_size, self._rnn_size }))
    else
      table.insert(inputs, prev_out)
    end
  end

  -- Remember inputs for the backward pass.
  if self.train then
    self.inputs[t] = inputs
  end

  local outputs = self:net(t):forward(inputs)
  local out = outputs[#outputs]
  local states = {}
  for i = 1, #outputs - 1 do
    table.insert(states, outputs[i])
  end

  return out, states
end

--[[Compute all forward steps.

  Parameters:

  * `batch` - based on data.lua
  * `encoder_states`
  * `context`
  * `func` - Calls `func(out, t)` each timestep.
--]]

function Decoder:forward_and_apply(batch, encoder_states, context, func)
  -- TODO: Make this a private method.

  if self.statesProto == nil then
    self.statesProto = utils.Tensor.initTensorTable(self._num_effective_layers,
                                                    self.stateProto,
                                                    { batch.size, self._rnn_size })
  end

  local states = utils.Tensor.copyTensorTable(self.statesProto, encoder_states)

  local prev_out

  for t = 1, batch.target_length do
    prev_out, states = self:forward_one(Data.get_target_input(batch, t), states, context, prev_out, t)
    func(prev_out, t)
  end
end

--[[Compute all forward steps.

Parameters:

  * `batch` - based on data.lua
  * `encoder_states` - the final encoder states
  * `context` - the context to apply attention to.

Returns: Tables of top hidden layer at each timestep.

--]]
function Decoder:forward(batch, encoder_states, context)
  if self.train then
    self.inputs = {}
  end

  local outputs = {}

  self:forward_and_apply(batch, encoder_states, context, function (out)
    table.insert(outputs, out)
  end)

  return outputs
end

--[[ Compute the standard backward update.

Parameters:

  * `batch`
  * `outputs`
  * `criterion`

  Note: This code is both the standard backward and criterion forward/backward.
  It returns both the gradInputs (ret 1 and 2) and the loss.
-- ]]
function Decoder:backward(batch, outputs, criterion)
  if self.gradOutputsProto == nil then
    self.gradOutputsProto = utils.Tensor.initTensorTable(self._num_effective_layers + 1,
                                                         self.gradOutputProto,
                                                         { batch.size, self._rnn_size })
  end

  local grad_states_input = utils.Tensor.reuseTensorTable(self.gradOutputsProto,
                                                          { batch.size, self._rnn_size })
  local grad_context_input = utils.Tensor.reuseTensor(self.gradContextProto,
                                                      { batch.size, batch.source_length, self._rnn_size })

  local grad_context_idx = #self.statesProto + 2
  local grad_input_feed_idx = #self.statesProto + 3

  local loss = 0

  for t = batch.target_length, 1, -1 do
    -- Compute decoder output gradients.
    -- Note: This would typically be in the forward pass.
    local pred = self.generator:forward(outputs[t])
    local output = Data.get_target_output(batch, t)

    loss = loss + criterion:forward(pred, output)

    -- Compute the criterion gradient.
    local gen_grad_out = criterion:backward(pred, output)
    for j = 1, #gen_grad_out do
      gen_grad_out[j]:div(batch.total_size)
    end

    -- Compute the final layer gradient.
    local dec_grad_out = self.generator:backward(outputs[t], gen_grad_out)
    grad_states_input[#grad_states_input]:add(dec_grad_out)

    -- Compute the standarad backward.
    local grad_input = self:net(t):backward(self.inputs[t], grad_states_input)

    -- Accumulate encoder output gradients.
    grad_context_input:add(grad_input[grad_context_idx])
    grad_states_input[#grad_states_input]:zero()

    -- Accumulate previous output gradients with input feeding gradients.
    if self._input_feed and t > 1 then
      grad_states_input[#grad_states_input]:add(grad_input[grad_input_feed_idx])
    end

    -- Prepare next decoder output gradients.
    for i = 1, #self.statesProto do
      grad_states_input[i]:copy(grad_input[i])
    end
  end

  return grad_states_input, grad_context_input, loss
end

--[[ Compute the loss on a batch based on final layer `generator`.]]
function Decoder:compute_loss(batch, encoder_states, context, criterion)
  local loss = 0
  self:forward_and_apply(batch, encoder_states, context, function (out, t)
    local pred = self.generator:forward(out)
    local output = Data.get_target_output(batch, t)
    loss = loss + criterion:forward(pred, output)
  end)

  return loss
end

--[[ Compute the cumulative score of a target sequence.
  Used in decoding when gold data are provided.
]]
function Decoder:compute_score(batch, encoder_states, context)
  local score = {}

  self:forward_and_apply(batch, encoder_states, context, function (out, t)
    local pred = self.generator:forward(out)
    for b = 1, batch.size do
      if t <= batch.target_size[b] then
        score[b] = (score[b] or 0) + pred[1][b][batch.target_output[t][b]]
      end
    end
  end)
  return score
end
