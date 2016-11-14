local model_utils = require 's2sa.utils.model_utils'
local Encoder = require 's2sa.encoder'
require 's2sa.model'

local BiEncoder, Model = torch.class('BiEncoder', 'Model')

function BiEncoder:__init(args, merge, net_fwd, net_bwd)
  Model.__init(self)

  -- preallocate full context vector
  self.context_proto = torch.zeros(args.max_batch_size, args.max_sent_length, args.rnn_size)

  -- preallocate full hidden states tensors
  self.states_proto = {}
  for _ = 1, args.num_layers do
    table.insert(self.states_proto, torch.zeros(args.max_batch_size, args.rnn_size))
    table.insert(self.states_proto, torch.zeros(args.max_batch_size, args.rnn_size))
  end

  if merge == 'concat' then
    if args.rnn_size % 2 ~= 0 then
      error('in concat mode, rnn_size must be divisible by 2')
    end
    args.rnn_size = args.rnn_size / 2
  elseif merge == 'sum' then
    args.rnn_size = args.rnn_size
  else
    error('invalid merge action ' .. merge)
  end

  self.merge = merge
  self.rnn_size = args.rnn_size

  self.fwd = Encoder.new(args, net_fwd)
  self.bwd = Encoder.new(args, net_bwd)
end

function BiEncoder:forward(batch)
  local fwd_states, fwd_context = self.fwd:forward(batch)

  -- reverse source input
  local tmp = batch.source_input
  batch.source_input = batch.source_input:index(1, torch.linspace(batch.source_length, 1, batch.source_length):long())

  local bwd_states, bwd_context = self.bwd:forward(batch)

  batch.source_input = tmp

  local states = model_utils.reset_state(self.states_proto, batch.size)
  local context = self.context_proto[{{1, batch.size}, {1, batch.source_length}}]

  if self.merge == 'concat' then
    context:narrow(3, 1, self.rnn_size):copy(fwd_context)
    context:narrow(3, self.rnn_size + 1, self.rnn_size):copy(bwd_context)

    for i = 1, #fwd_states do
      states[i]:narrow(2, 1, self.rnn_size):copy(fwd_states[i])
      states[i]:narrow(2, self.rnn_size + 1, self.rnn_size):copy(bwd_states[i])
    end
  elseif self.merge == 'sum' then
    for i = 1, #states do
      states[i]:copy(fwd_states[i])
      states[i]:add(bwd_states[i])
    end
    for t = 1, batch.source_length do
      context[{{}, t}]:copy(fwd_context[{{}, t}])
      context[{{}, t}]:add(bwd_context[{{}, t}])
    end
  end

  return states, context
end

function BiEncoder:backward(batch, grad_states_output, grad_context_output)
  if self.merge == 'concat' then
    local grad_context_output_split = grad_context_output:chunk(2, 3)
    local grad_context_output_fwd = grad_context_output_split[1]
    local grad_context_output_bwd = grad_context_output_split[2]

    local grad_states_output_fwd = {}
    local grad_states_output_bwd = {}

    for i = 1, #grad_states_output do
      local states_split = grad_states_output[i]:chunk(2, 2)
      table.insert(grad_states_output_fwd, states_split[1])
      table.insert(grad_states_output_bwd, states_split[2])
    end

    self.fwd:backward(batch, grad_states_output_fwd, grad_context_output_fwd)
    self.bwd:backward(batch, grad_states_output_bwd, grad_context_output_bwd)
  elseif self.merge == 'sum' then
    self.fwd:backward(batch, grad_states_output, grad_context_output)
    self.bwd:backward(batch, grad_states_output, grad_context_output)
  end
end

function BiEncoder:training()
  self.fwd:training()
  self.bwd:training()
end

function BiEncoder:evaluate()
  self.fwd:evaluate()
  self.bwd:evaluate()
end

function BiEncoder:convert(f)
  self.fwd:convert(f)
  self.bwd:convert(f)

  self.context_proto = f(self.context_proto)
  for i = 1, #self.states_proto do
    self.states_proto[i] = f(self.states_proto[i])
  end
end

return BiEncoder
