local Encoder = require 's2sa.encoder'

local BiEncoder = torch.class('BiEncoder')

function BiEncoder:__init(args, merge, net_fwd, net_bwd)
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

  local states
  local context

  if self.merge == 'concat' then
    states = {}
    context = torch.cat(fwd_context, bwd_context, 3)

    for i = 1, #fwd_states do
      table.insert(states, torch.cat(fwd_states[i], bwd_states[i]))
    end
  elseif self.merge == 'sum' then
    states = fwd_states
    context = fwd_context

    for i = 1, #states do
      states[i]:add(bwd_states[i])
    end
    for t = 1, batch.source_length do
      context[{{}, t}]:add(bwd_context[{{}, t}])
    end
  end

  return states, context
end

function BiEncoder:backward(batch, grad_states_output, grad_context_output)
  if self.merge == 'concat' then
    local grad_context_output_split = grad_context_output:split(self.rnn_size, 3)
    local grad_context_output_fwd = grad_context_output_split[1]
    local grad_context_output_bwd = grad_context_output_split[2]

    local grad_states_output_fwd = {}
    local grad_states_output_bwd = {}

    for i = 1, #grad_states_output do
      local states_split = grad_states_output[i]:split(self.rnn_size, 2)
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

function BiEncoder:float()
  self.fwd:float()
  self.bwd:float()
end

function BiEncoder:double()
  self.fwd:double()
  self.bwd:double()
end

function BiEncoder:cuda()
  self.fwd:cuda()
  self.bwd:cuda()
end

return BiEncoder
