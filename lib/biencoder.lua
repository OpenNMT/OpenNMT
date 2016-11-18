local model_utils = require 'lib.utils.model_utils'
local Encoder = require 'lib.encoder'
require 'lib.model'

local function reverse_input(batch)
  batch.source_input, batch.source_input_rev = batch.source_input_rev, batch.source_input
  batch.source_input_pad_left, batch.source_input_rev_pad_left = batch.source_input_rev_pad_left, batch.source_input_pad_left
end


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

  -- preallocate gradient of the backward context
  if args.training then
    self.grad_context_bwd_proto = torch.zeros(args.max_batch_size, args.max_sent_length, args.rnn_size)
  end

  self.merge = merge
  self.rnn_size = args.rnn_size

  self.fwd = Encoder.new(args, net_fwd)
  self.bwd = Encoder.new(args, net_bwd)
end

function BiEncoder:resize_proto(batch_size)
  self.context_proto:resize(batch_size, self.context_proto:size(2), self.context_proto:size(3))
  for i = 1, #self.states_proto do
    self.states_proto[i]:resize(batch_size, self.states_proto[i]:size(2))
  end
end

function BiEncoder:forward(batch)
  local fwd_states, fwd_context = self.fwd:forward(batch)

  reverse_input(batch)
  local bwd_states, bwd_context = self.bwd:forward(batch)
  reverse_input(batch)

  local states = model_utils.reset_state(self.states_proto, batch.size)
  local context = self.context_proto[{{1, batch.size}, {1, batch.source_length}}]

  if self.merge == 'concat' then
    for i = 1, #fwd_states do
      states[i]:narrow(2, 1, self.rnn_size):copy(fwd_states[i])
      states[i]:narrow(2, self.rnn_size + 1, self.rnn_size):copy(bwd_states[i])
    end
    for t = 1, batch.source_length do
      context[{{}, t}]:narrow(2, 1, self.rnn_size)
        :copy(fwd_context[{{}, t}])
      context[{{}, t}]:narrow(2, self.rnn_size + 1, self.rnn_size)
        :copy(bwd_context[{{}, batch.source_length - t + 1}])
    end
  elseif self.merge == 'sum' then
    for i = 1, #states do
      states[i]:copy(fwd_states[i])
      states[i]:add(bwd_states[i])
    end
    for t = 1, batch.source_length do
      context[{{}, t}]:copy(fwd_context[{{}, t}])
      context[{{}, t}]:add(bwd_context[{{}, batch.source_length - t + 1}])
    end
  end

  return states, context
end

function BiEncoder:backward(batch, grad_states_output, grad_context_output)
  local grad_context_output_fwd
  local grad_context_output_bwd

  local grad_states_output_fwd = {}
  local grad_states_output_bwd = {}

  if self.merge == 'concat' then
    local grad_context_output_split = grad_context_output:chunk(2, 3)
    grad_context_output_fwd = grad_context_output_split[1]
    grad_context_output_bwd = grad_context_output_split[2]

    for i = 1, #grad_states_output do
      local states_split = grad_states_output[i]:chunk(2, 2)
      table.insert(grad_states_output_fwd, states_split[1])
      table.insert(grad_states_output_bwd, states_split[2])
    end
  elseif self.merge == 'sum' then
    grad_context_output_fwd = grad_context_output
    grad_context_output_bwd = grad_context_output

    grad_states_output_fwd = grad_states_output
    grad_states_output_bwd = grad_states_output
  end

  self.fwd:backward(batch, grad_states_output_fwd, grad_context_output_fwd)

  -- reverse gradients of the backward context
  local grad_context_bwd = self.grad_context_bwd_proto[{{1, batch.size}, {1, batch.source_length}}]

  for t = 1, batch.source_length do
    grad_context_bwd[{{}, t}]:copy(grad_context_output_bwd[{{}, batch.source_length - t + 1}])
  end

  self.bwd:backward(batch, grad_states_output_bwd, grad_context_bwd)
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

  if self.grad_context_bwd_proto ~= nil then
    self.grad_context_bwd_proto = f(self.grad_context_bwd_proto)
  end

  for i = 1, #self.states_proto do
    self.states_proto[i] = f(self.states_proto[i])
  end
end

return BiEncoder
