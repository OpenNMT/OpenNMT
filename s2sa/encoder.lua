local model_utils = require 's2sa.model_utils'
local table_utils = require 's2sa.table_utils'
local cuda = require 's2sa.cuda'
require 's2sa.sequencer'

local Encoder, Sequencer = torch.class('Encoder', 'Sequencer')

function Encoder:__init(args, opt)
  self.context_proto = cuda.convert(torch.zeros(opt.max_batch_size, args.max_sent_length, opt.rnn_size))

  Sequencer.__init(self, args, opt, 'enc')
end

function Encoder:forward(batch)
  local states = model_utils.reset_state(self.init_states, batch.size)
  local context = self.context_proto[{{1, batch.size}, {1, batch.source_length}}]

  self.inputs = {}

  for t = 1, batch.source_length do
    self.inputs[t] = {}
    table_utils.append(self.inputs[t], states)
    table_utils.append(self.inputs[t], {batch.source_input[t]})

    states = Sequencer.get_clone(self, t):forward(self.inputs[t])

    context[{{}, t}]:copy(states[#states])
  end

  return states, context
end

function Encoder:backward(batch, grad_states_output, grad_context_output)
  local grad_states_input = table_utils.clone(grad_states_output)

  for t = batch.source_length, 1, -1 do
    -- add context gradients to last hidden states gradients
    grad_states_input[#grad_states_input]:add(grad_context_output[{{}, t}])

    local grad_input = self.network_clones[t]:backward(self.inputs[t], grad_states_input)

    -- prepare next encoder output gradients
    for i = 1, #grad_states_input do
      grad_states_input[i]:copy(grad_input[i])
    end
  end

  Sequencer.backward_word_vecs(self)
end

function Encoder:cuda()
  Sequencer.cuda(self)
  self.context_proto = self.context_proto:cuda()
end

return Encoder
