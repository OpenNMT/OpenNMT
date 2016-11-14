local model_utils = require 's2sa.utils.model_utils'
local table_utils = require 's2sa.utils.table_utils'
local cuda = require 's2sa.utils.cuda'
require 's2sa.sequencer'

local Encoder, Sequencer = torch.class('Encoder', 'Sequencer')

function Encoder:__init(args, network)
  Sequencer.__init(self, 'enc', args, network)
  self.mask_padding = args.mask_padding or false

  -- preallocate context vector
  self.context_proto = torch.zeros(args.max_batch_size, args.max_sent_length, args.rnn_size)

  if args.training then
    -- preallocate output gradients
    self.grad_out_proto = {}
    for _ = 1, args.num_layers do
      table.insert(self.grad_out_proto, torch.zeros(args.max_batch_size, args.rnn_size))
      table.insert(self.grad_out_proto, torch.zeros(args.max_batch_size, args.rnn_size))
    end
  end
end

function Encoder:forward(batch)
  local states = model_utils.reset_state(self.states_proto, batch.size)
  local context = self.context_proto[{{1, batch.size}, {1, batch.source_length}}]

  if not self.eval_mode then
    self.inputs = {}
  end

  for t = 1, batch.source_length do
    local inputs = {}
    table_utils.append(inputs, states)
    table.insert(inputs, batch.source_input[t])

    if not self.eval_mode then
      -- remember inputs for the backward pass
      self.inputs[t] = inputs
    end

    states = Sequencer.net(self, t):forward(inputs)

<<<<<<< Updated upstream
=======
    if self.mask_padding then
      for b = 1, batch.size do
        if t <= batch.source_length - batch.source_size[b] then
          for j = 1, #states do
            states[j][b]:zero()
          end
        end
      end
    end

>>>>>>> Stashed changes
    context[{{}, t}]:copy(states[#states])
  end

  return states, context
end

function Encoder:backward(batch, grad_states_output, grad_context_output)
  local grad_states_input = model_utils.copy_state(self.grad_out_proto, grad_states_output, batch.size)

  for t = batch.source_length, 1, -1 do
    -- add context gradients to last hidden states gradients
    grad_states_input[#grad_states_input]:add(grad_context_output[{{}, t}])

    local grad_input = Sequencer.net(self, t):backward(self.inputs[t], grad_states_input)

    -- prepare next encoder output gradients
    for i = 1, #grad_states_input do
      grad_states_input[i]:copy(grad_input[i])
    end
  end

  Sequencer.backward_word_vecs(self)
end

function Encoder:convert(f)
  Sequencer.convert(self, f)
  self.context_proto = f(self.context_proto)
end

return Encoder
