local model_utils = require 's2sa.utils.model_utils'
local table_utils = require 's2sa.utils.table_utils'
local cuda = require 's2sa.utils.cuda'
require 's2sa.sequencer'

local Decoder, Sequencer = torch.class('Decoder', 'Sequencer')

function Decoder:__init(args, network)
  Sequencer.__init(self, 'dec', args, network)
  self.input_feed = args.input_feed

  -- preallocate output gradients
  self.grad_out_proto = {}
  for _ = 1, args.num_layers do
    table.insert(self.grad_out_proto, torch.zeros(args.max_batch_size, args.rnn_size))
    table.insert(self.grad_out_proto, torch.zeros(args.max_batch_size, args.rnn_size))
  end
  table.insert(self.grad_out_proto, torch.zeros(args.max_batch_size, args.rnn_size))

  -- preallocate context gradient
  self.grad_context_proto = torch.zeros(args.max_batch_size, args.max_source_length, args.rnn_size)

  -- preallocate default input feeding tensor
  if self.input_feed then
    self.input_feed_proto = torch.zeros(args.max_batch_size, args.rnn_size)
  end
end

function Decoder:forward(batch, encoder_states, context)
  local states = model_utils.copy_state(self.states_proto, encoder_states, batch.size)

  self.inputs = {}
  local outputs = {}

  for t = 1, batch.target_length do
    self.inputs[t] = {}
    table_utils.append(self.inputs[t], states)
    table_utils.append(self.inputs[t], {batch.target_input[t]})
    table_utils.append(self.inputs[t], {context})
    if self.input_feed then
      if #outputs == 0 then
        table_utils.append(self.inputs[t], {self.input_feed_proto[{{1, batch.size}}]})
      else
        table_utils.append(self.inputs[t], {outputs[#outputs]})
      end
    end

    local out = Sequencer.net(self, t):forward(self.inputs[t])

    -- store attention layer output
    table.insert(outputs, out[#out])

    states = {}
    for i = 1, #out - 1 do
      table.insert(states, out[i])
    end
  end

  return outputs
end

function Decoder:backward(batch, outputs, generator)
  local grad_states_input = model_utils.reset_state(self.grad_out_proto, batch.size)
  local grad_context_input = self.grad_context_proto[{{1, batch.size}, {1, batch.source_length}}]:zero()

  local grad_context_idx = #self.states_proto + 2
  local grad_input_feed_idx = #self.states_proto + 3

  local loss = 0

  for t = batch.target_length, 1, -1 do
    -- compute decoder output gradients
    local pred = generator.network:forward(outputs[t])
    loss = loss + generator.criterion:forward(pred, batch.target_output[t]) / batch.size
    local gen_grad_out = generator.criterion:backward(pred, batch.target_output[t])
    gen_grad_out:div(batch.size)
    local dec_grad_out = generator.network:backward(outputs[t], gen_grad_out)
    grad_states_input[#grad_states_input]:add(dec_grad_out)

    local grad_input = Sequencer.net(self, t):backward(self.inputs[t], grad_states_input)

    -- accumulate encoder output gradients
    grad_context_input:add(grad_input[grad_context_idx])

    grad_states_input[#grad_states_input]:zero()

    -- accumulate previous output gradients with input feeding gradients
    if self.input_feed and t > 1 then
      grad_states_input[#grad_states_input]:add(grad_input[grad_input_feed_idx])
    end

    -- prepare next decoder output gradients
    for i = 1, #self.states_proto do
      grad_states_input[i]:copy(grad_input[i])
    end
  end

  Sequencer.backward_word_vecs(self)

  return grad_states_input, grad_context_input, loss
end

function Decoder:convert(f)
  Sequencer.convert(self, f)
  self.input_feed_proto = f(self.input_feed_proto)
  self.grad_context_proto = f(self.grad_context_proto)
end

return Decoder
