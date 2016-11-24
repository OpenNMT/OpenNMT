local ModelUtils = require 'lib.utils.model_utils'
local table_utils = require 'lib.utils.table_utils'
local cuda = require 'lib.utils.cuda'
require 'lib.sequencer'

require 'lib.MaskedSoftmax'

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
local Decoder, Sequencer = torch.class('Decoder', 'Sequencer')

function Decoder:__init(args, network, generator)
  Sequencer.__init(self, 'dec', args, network)

  -- The generator use the output of the decoder sequencer to generate the
  -- likelihoods over the target vocabulary.
  self.generator = generator or self:_buildGenerator(args.vocab_size, args.rnn_size)

  -- Input feeding means the decoder takes an extra
  -- vector each time representing the attention at the
  -- previous step.
  self.input_feed = args.input_feed
  if self.input_feed then
    self.inputFeedProto = torch.Tensor()
  end

  -- Mask padding means that the attention-layer is constrained to
  -- give zero-weight to padding. This is done by storing a reference
  -- to the softmax attention-layer.
  self.mask_padding = args.mask_padding or false
  if self.mask_padding then
    self.network:apply(function (layer)
      if layer.name == 'decoder_attn' then
        self.decoder_attn = layer
      end
    end)
  end

  -- Prototype for preallocated context gradient.
  self.gradContextProto = torch.Tensor()
end

function Decoder:_buildGenerator(vocab_size, rnn_size)
  -- Builds a layer for predicting words.
  -- Layer used maps from (h^L) => (V).
  local inputs = {}
  table.insert(inputs, nn.Identity()())

  local map = nn.Linear(rnn_size, vocab_size)(inputs[1])

  -- Use cudnn logsoftmax if available.
  local loglk = cuda.nn.LogSoftMax()(map)

  return nn.gModule(inputs, {loglk})
end

--[[ Update internals to prepare for new batch.]]
function Decoder:reset(source_sizes, source_length, beam_size)
  self.decoder_attn:replace(function(module)
    if module.name == 'softmax_attn' then
      local mod
      if source_sizes ~= nil then
        mod = MaskedSoftmax(source_sizes, source_length, beam_size)
      else
        mod = nn.SoftMax()
      end

      mod.name = 'softmax_attn'
      mod = cuda.convert(mod)
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
  table_utils.append(inputs, prev_states)
  table.insert(inputs, input)
  table.insert(inputs, context)
  if self.input_feed then
    if prev_out == nil then
      table.insert(inputs, ModelUtils.reuseTensor(self.inputFeedProto,
                                                  { input:size(1), self.args.rnn_size }))
    else
      table.insert(inputs, prev_out)
    end
  end

  -- Remember inputs for the backward pass.
  if self.train then
    self.inputs[t] = inputs
  end

  -- TODO: self:net?
  local outputs = Sequencer.net(self, t):forward(inputs)
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
    self.statesProto = ModelUtils.initTensorTable(self.args.num_layers * 2,
                                                  self.stateProto,
                                                  { batch.size, self.args.rnn_size })
  end

  local states = ModelUtils.copyTensorTable(self.statesProto, encoder_states)

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
    self.gradOutputsProto = ModelUtils.initTensorTable(self.args.num_layers * 2 + 1,
                                                       self.gradOutputProto,
                                                       { batch.size, self.args.rnn_size })
  end

  local grad_states_input = ModelUtils.reuseTensorTable(self.gradOutputsProto,
                                                        { batch.size, self.args.rnn_size })
  local grad_context_input = ModelUtils.reuseTensor(self.gradContextProto,
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
    local grad_input = Sequencer.net(self, t):backward(self.inputs[t], grad_states_input)

    -- Accumulate encoder output gradients.
    grad_context_input:add(grad_input[grad_context_idx])
    grad_states_input[#grad_states_input]:zero()

    -- Accumulate previous output gradients with input feeding gradients.
    if self.input_feed and t > 1 then
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
