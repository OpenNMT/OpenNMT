local model_utils = require 'lib.utils.model_utils'
local table_utils = require 'lib.utils.table_utils'
local cuda = require 'lib.utils.cuda'
require 'lib.sequencer'

require 'lib.MaskedSoftmax'

--[[ Decoder is the sequencer for the target words.]]
local Decoder, Sequencer = torch.class('Decoder', 'Sequencer')

function Decoder:__init(args, network)
  Sequencer.__init(self, 'dec', args, network)
  -- Input feeding means the decoder takes an extra
  -- vector each time representing the attention at the
  -- previous step.
  self.input_feed = args.input_feed
  if self.input_feed then
    self.input_feed_proto = torch.zeros(args.max_batch_size, args.rnn_size)
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


  if args.training then
    -- Preallocate output gradients and context gradient.
    -- TODO: move all this to sequencer.
    self.grad_out_proto = {}
    for _ = 1, args.num_layers do
      table.insert(self.grad_out_proto, torch.zeros(args.max_batch_size, args.rnn_size))
      table.insert(self.grad_out_proto, torch.zeros(args.max_batch_size, args.rnn_size))
    end
    table.insert(self.grad_out_proto, torch.zeros(args.max_batch_size, args.rnn_size))

    self.grad_context_proto = torch.zeros(args.max_batch_size, args.max_source_length, args.rnn_size)
  end
end

--[[ Call to change the `batch_size`.

  TODO: rename.
]]
function Decoder:resize_proto(batch_size)
  Sequencer.resize_proto(self, batch_size)
  if self.input_feed then
    self.input_feed_proto:resize(batch_size, self.input_feed_proto:size(2)):zero()
  end
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
      table.insert(inputs, self.input_feed_proto[{{1, input:size(1)}}])
    else
      table.insert(inputs, prev_out)
    end
  end

  -- Remember inputs for the backward pass.
  if not self.eval_mode then
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
  local states = model_utils.copy_state(self.states_proto, encoder_states, batch.size)

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
  if not self.eval_mode then
    self.inputs = {}
  end

  local outputs = {}

  self:forward_and_apply(batch, encoder_states, context, function (out)
    table.insert(outputs, out)
  end)

  return outputs
end

function Decoder:compute_score(batch, encoder_states, context, generator)
  -- TODO: Why do we need this method?
  local score = {}

  self:forward_and_apply(batch, encoder_states, context, function (out, t)
    local pred = generator.network:forward(out)
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
function Decoder:compute_loss(batch, encoder_states, context, generator)

  local loss = 0
  self:forward_and_apply(batch, encoder_states, context, function (out, t)
    local pred = generator.network:forward(out)
    loss = loss + generator.criterion:forward(pred, batch.target_output[t])
  end)

  return loss
end

--[[ Compute the standard backward update.
  With input `batch`, target `outputs`, and `generator`
  Note: This code is both the standard backward and criterion forward/backward.
  It returns both the gradInputs (ret 1 and 2) and the loss.

  TODO: This object should own `generator` and or, generator should
  control its own backward pass.
]]
function Decoder:backward(batch, outputs, generator)
  local grad_states_input = model_utils.reset_state(self.grad_out_proto, batch.size)
  local grad_context_input = self.grad_context_proto[{{1, batch.size}, {1, batch.source_length}}]:zero()

  local grad_context_idx = #self.states_proto + 2
  local grad_input_feed_idx = #self.states_proto + 3

  local loss = 0

  for t = batch.target_length, 1, -1 do
    -- Compute decoder output gradients.
    -- Note: This would typically be in the forward pass.
    local pred = generator.network:forward(outputs[t])
    loss = loss + generator.criterion:forward(pred, batch.target_output[t]) / batch.size


    -- Compute the criterion gradient.
    local gen_grad_out = generator.criterion:backward(pred, batch.target_output[t])
    gen_grad_out:div(batch.size)

    -- Compute the final layer gradient.
    local dec_grad_out = generator.network:backward(outputs[t], gen_grad_out)
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
    for i = 1, #self.states_proto do
      grad_states_input[i]:copy(grad_input[i])
    end
  end

  return grad_states_input, grad_context_input, loss
end

function Decoder:convert(f)
  -- TODO: move up to sequencer.
  Sequencer.convert(self, f)
  self.input_feed_proto = f(self.input_feed_proto)

  if self.grad_context_proto ~= nil then
    self.grad_context_proto = f(self.grad_context_proto)
  end
end

return Decoder
