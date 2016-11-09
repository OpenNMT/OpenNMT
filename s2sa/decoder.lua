local model_utils = require 's2sa.utils.model_utils'
local table_utils = require 's2sa.utils.table_utils'
local cuda = require 's2sa.utils.cuda'
require 's2sa.sequencer'

local Decoder, Sequencer = torch.class('Decoder', 'Sequencer')

function Decoder:__init(args, network)
  Sequencer.__init(self, 'dec', args, network)

  self.input_feed = args.input_feed
  if self.input_feed then
    self.input_feed_proto = cuda.convert(torch.zeros(args.max_batch_size, args.rnn_size))
  end
end

function Decoder:forward(batch, encoder_states, context)
  local states = encoder_states

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

    states = Sequencer.get_clone(self, t):forward(self.inputs[t])

    -- store attention layer output
    table.insert(outputs, states[#states])
    table.remove(states)
  end

  return outputs
end

function Decoder:backward(batch, grad_output)
  local grad_states_input = model_utils.reset_state(self.init_states, batch.size)
  local grad_context_input = nil
  local grad_context_idx = #grad_states_input + 2

  for t = batch.target_length, 1, -1 do
    table.insert(grad_states_input, grad_output[t])
    local grad_input = self.network_clones[t]:backward(self.inputs[t], grad_states_input)
    table.remove(grad_states_input)

    -- prepare next decoder output gradients
    for i = 1, #grad_states_input do
      grad_states_input[i] = grad_input[i]
    end

    -- accumulate encoder output gradients
    if grad_context_input == nil then
      grad_context_input = grad_input[grad_context_idx]
    else
      grad_context_input:add(grad_input[grad_context_idx])
    end

    -- accumulate previous output gradients with input feeding gradients
    if self.input_feed and t > 1 then
      grad_output[t - 1]:add(grad_input[#grad_input])
    end
  end

  Sequencer.backward_word_vecs(self)

  return grad_states_input, grad_context_input
end

return Decoder
