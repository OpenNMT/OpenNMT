local model_utils = require 's2sa.model_utils'
local table_utils = require 's2sa.table_utils'
require 's2sa.sequencer'

local Decoder, Sequencer = torch.class('Decoder', 'Sequencer')

function Decoder:__init(args, opt)
  Sequencer.__init(self, args, opt, 'dec')
end

function Decoder:forward(batch, encoder_states, context)
  local states = table_utils.clone(encoder_states)

  self.inputs = {}
  local outputs = {}

  for t = 1, batch.target_length do
    self.inputs[t] = {}
    table_utils.append(self.inputs[t], states)
    table_utils.append(self.inputs[t], {batch.target_input[t]})
    table_utils.append(self.inputs[t], {context})

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

  for t = batch.target_length, 1, -1 do
    table.insert(grad_states_input, grad_output[t])
    local grad_input = self.network_clones[t]:backward(self.inputs[t], grad_states_input)
    table.remove(grad_states_input)

    -- prepare next decoder output gradients
    for i = 1, #grad_states_input do
      grad_states_input[i]:copy(grad_input[i])
    end

    -- accumulate encoder output gradients
    if grad_context_input == nil then
      grad_context_input = grad_input[#grad_input]:clone()
    else
      grad_context_input:add(grad_input[#grad_input])
    end
  end

  Sequencer.backward_word_vecs(self)

  return grad_states_input, grad_context_input
end

return Decoder
