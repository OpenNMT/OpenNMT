--[[ Decoder is the sequencer for the target words.

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

Inherits from [Sequencer](lib+sequencer).

--]]
local Decoder, Sequencer = torch.class('onmt.Decoder', 'onmt.Sequencer')

function Decoder:__init(args, network, generator)
  Sequencer.__init(self, args, network or self:_buildModel(args))

  -- The generator use the output of the decoder sequencer to generate the
  -- likelihoods over the target vocabulary.
  self.generator = generator or self:_buildGenerator(args.vocab_size, args.rnn_size)

  -- Input feeding means the decoder takes an extra
  -- vector each time representing the attention at the
  -- previous step.
  if self.args.input_feed then
    self.inputFeedProto = torch.Tensor()
  end

  -- Mask padding means that the attention-layer is constrained to
  -- give zero-weight to padding. This is done by storing a reference
  -- to the softmax attention-layer.
  if self.args.mask_padding then
    self.network:apply(function (layer)
      if layer.name == 'decoder_attn' then
        self.decoder_attn = layer
      end
    end)
  end

  -- Prototype for preallocated context gradient.
  self.gradContextProto = torch.Tensor()
end

--[[ Build one time-step of a decoder

Parameters:

  * `args` - global args.

Returns: An nn-graph mapping

  $${(c^1_{t-1}, h^1_{t-1}, .., c^L_{t-1}, h^L_{t-1}, x_t, con/H, if) =>
  (c^1_{t}, h^1_{t}, .., c^L_{t}, h^L_{t}, a)}$$

  Where ${c^l}$ and ${h^l}$ are the hidden and cell states at each layer,
  ${x_t}$ is a sparse word to lookup,
  ${con/H}$ is the context/source hidden states for attention,
  ${if}$ is the input feeding, and
  ${a}$ is the context vector computed at this timestep.
--]]
function Decoder:_buildModel(args)
  local inputs = {}
  local states = {}

  -- Inputs are previous layers first.
  for _ = 1, args.num_layers do
    local c0 = nn.Identity()() -- batch_size x rnn_size
    table.insert(inputs, c0)
    table.insert(states, c0)

    local h0 = nn.Identity()() -- batch_size x rnn_size
    table.insert(inputs, h0)
    table.insert(states, h0)
  end

  local x = nn.Identity()() -- batch_size
  table.insert(inputs, x)

  local context = nn.Identity()() -- batch_size x source_length x rnn_size
  table.insert(inputs, context)

  local input_feed
  if args.input_feed then
    input_feed = nn.Identity()() -- batch_size x rnn_size
    table.insert(inputs, input_feed)
  end

  local input_size = args.word_vec_size

  -- Compute word embedding.
  local input = onmt.WordEmbedding(args.vocab_size, input_size, args.pre_word_vecs, args.fix_word_vecs)(x)

  -- If set, concatenate previous decoder output.
  if args.input_feed then
    input = nn.JoinTable(2)({input, input_feed})
    input_size = input_size + args.rnn_size
  end

  table.insert(states, input)

  -- Forward states and input into the RNN.
  local outputs = onmt.LSTM(args.num_layers, input_size, args.rnn_size, args.dropout)(states)

  -- The output of a subgraph is a node: split it to access the last RNN output.
  outputs = { outputs:split(args.num_layers * 2) }

  -- Compute the attention here using h^L as query.
  local attn_layer = onmt.GlobalAttention(args.rnn_size)
  attn_layer.name = 'decoder_attn'
  local attn_output = attn_layer({outputs[#outputs], context})
  if args.dropout > 0 then
    attn_output = nn.Dropout(args.dropout)(attn_output)
  end
  table.insert(outputs, attn_output)

  return nn.gModule(inputs, outputs)
end

function Decoder:_buildGenerator(vocab_size, rnn_size)
  -- Builds a layer for predicting words.
  -- Layer used maps from (h^L) => (V).
  local inputs = {}
  table.insert(inputs, nn.Identity()())

  local map = nn.Linear(rnn_size, vocab_size)(inputs[1])

  -- Use cudnn logsoftmax if available.
  local loglk = utils.Cuda.nn.LogSoftMax()(map)

  return nn.gModule(inputs, {loglk})
end

--[[ Update internals to prepare for new batch.]]
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
  if self.args.input_feed then
    if prev_out == nil then
      table.insert(inputs, utils.Model.reuseTensor(self.inputFeedProto,
                                                   { input:size(1), self.args.rnn_size }))
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
  if self.statesProto == nil then
    self.statesProto = utils.Model.initTensorTable(self.args.num_layers * 2,
                                                   self.stateProto,
                                                   { batch.size, self.args.rnn_size })
  end

  local states = utils.Model.copyTensorTable(self.statesProto, encoder_states)

  local prev_out

  for t = 1, batch.target_length do
    prev_out, states = self:forward_one(batch.target_input[t], states, context, prev_out, t)
    func(prev_out, t)
  end
end

--[[Compute all forward steps.

Parameters:

  * `batch` - based on data.lua
  * `encoder_states`
  * `context`

Returns:

  1. `outputs` - Top Hidden layer at each time-step.
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

function Decoder:compute_score(batch, encoder_states, context)
  -- TODO: Why do we need this method?
  local score = {}

  self:forward_and_apply(batch, encoder_states, context, function (out, t)
    local pred = self.generator:forward(out)
    for b = 1, batch.size do
      if t <= batch.target_size[b] then
        score[b] = score[b] or 0
        score[b] = score[b] + pred[b][batch.target_output[t][b]]
      end
    end
  end)

  return score
end

--[[ Compute the loss on a batch based on final layer `generator`.]]
function Decoder:compute_loss(batch, encoder_states, context, criterion)

  local loss = 0
  self:forward_and_apply(batch, encoder_states, context, function (out, t)
    local pred = self.generator:forward(out)
    loss = loss + criterion:forward(pred, batch.target_output[t])
  end)

  return loss
end

--[[ Compute the standard backward update.
  With input `batch`, target `outputs`, and `criterion`
  Note: This code is both the standard backward and criterion forward/backward.
  It returns both the gradInputs (ret 1 and 2) and the loss.
]]
function Decoder:backward(batch, outputs, criterion)
  if self.gradOutputsProto == nil then
    self.gradOutputsProto = utils.Model.initTensorTable(self.args.num_layers * 2 + 1,
                                                        self.gradOutputProto,
                                                        { batch.size, self.args.rnn_size })
  end

  local grad_states_input = utils.Model.reuseTensorTable(self.gradOutputsProto,
                                                         { batch.size, self.args.rnn_size })
  local grad_context_input = utils.Model.reuseTensor(self.gradContextProto,
                                                     { batch.size, batch.source_length, self.args.rnn_size })

  local grad_context_idx = #self.statesProto + 2
  local grad_input_feed_idx = #self.statesProto + 3

  local loss = 0

  for t = batch.target_length, 1, -1 do
    -- Compute decoder output gradients.
    -- Note: This would typically be in the forward pass.
    local pred = self.generator:forward(outputs[t])
    loss = loss + criterion:forward(pred, batch.target_output[t]) / batch.size


    -- Compute the criterion gradient.
    local gen_grad_out = criterion:backward(pred, batch.target_output[t])
    gen_grad_out:div(batch.size)

    -- Compute the final layer gradient.
    local dec_grad_out = self.generator:backward(outputs[t], gen_grad_out)
    grad_states_input[#grad_states_input]:add(dec_grad_out)

    -- Compute the standarad backward.
    local grad_input = self:net(t):backward(self.inputs[t], grad_states_input)

    -- Accumulate encoder output gradients.
    grad_context_input:add(grad_input[grad_context_idx])
    grad_states_input[#grad_states_input]:zero()

    -- Accumulate previous output gradients with input feeding gradients.
    if self.args.input_feed and t > 1 then
      grad_states_input[#grad_states_input]:add(grad_input[grad_input_feed_idx])
    end

    -- Prepare next decoder output gradients.
    for i = 1, #self.statesProto do
      grad_states_input[i]:copy(grad_input[i])
    end
  end

  return grad_states_input, grad_context_input, loss
end

function Decoder:training()
  Sequencer.training(self)
  self.generator:training()
end

function Decoder:evaluate()
  Sequencer.evaluate(self)
  self.generator:evaluate()
end

return Decoder
