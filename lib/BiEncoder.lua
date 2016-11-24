local ModelUtils = require 'lib.utils.model_utils'
local Encoder = require 'lib.Encoder'

local function reverse_input(batch)
  batch.source_input, batch.source_input_rev = batch.source_input_rev, batch.source_input
  batch.source_input_pad_left, batch.source_input_rev_pad_left = batch.source_input_rev_pad_left, batch.source_input_pad_left
end

--[[ BiEncoder is a bidirectional Sequencer used for the source language.


* `net_fwd`

    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     .      .      .             .
     |      |      |             |
    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     |      |      |             |
    x_1    x_2    x_3           x_n

* `net_bwd`

    h_1 <= h_2 <= h_3 <= ... <= h_n
     |      |      |             |
     .      .      .             .
     |      |      |             |
    h_1 <= h_2 <= h_3 <= ... <= h_n
     |      |      |             |
     |      |      |             |
    x_1    x_2    x_3           x_n

--]]

local BiEncoder, parent = torch.class('BiEncoder', 'nn.Module')

--[[ Creates two Encoder's (encoder.lua) `net_fwd` and `net_bwd`.
  The two are combined use `merge` operation (concat/sum).
]]
function BiEncoder:__init(args, merge, net_fwd, net_bwd)
  parent.__init(self)

  -- Prototype for preallocated full context vector.
  self.contextProto = torch.Tensor()

  -- Prototype for preallocated full hidden states tensors.
  self.stateProto = torch.Tensor()

  -- Prototype for preallocated gradient of the backward context
  self.gradContextBwdProto = torch.Tensor()

  self.rnn_size = args.rnn_size

  -- Comput the merge operation.
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
  self.args = args

  self.fwd = Encoder.new(args, net_fwd)
  self.bwd = Encoder.new(args, net_bwd)
end

function BiEncoder:forward(batch)
  local fwd_states, fwd_context = self.fwd:forward(batch)

  reverse_input(batch)
  local bwd_states, bwd_context = self.bwd:forward(batch)
  reverse_input(batch)

  if self.statesProto == nil then
    self.statesProto = ModelUtils.initTensorTable(self.args.num_layers * 2,
                                                  self.stateProto,
                                                  { batch.size, self.rnn_size })
  end

  local states = ModelUtils.reuseTensorTable(self.statesProto, { batch.size, self.rnn_size })
  local context = ModelUtils.reuseTensor(self.contextProto,
                                         { batch.size, batch.source_length, self.rnn_size })

  if self.merge == 'concat' then
    for i = 1, #fwd_states do
      states[i]:narrow(2, 1, self.args.rnn_size):copy(fwd_states[i])
      states[i]:narrow(2, self.args.rnn_size + 1, self.args.rnn_size):copy(bwd_states[i])
    end
    for t = 1, batch.source_length do
      context[{{}, t}]:narrow(2, 1, self.args.rnn_size)
        :copy(fwd_context[{{}, t}])
      context[{{}, t}]:narrow(2, self.args.rnn_size + 1, self.args.rnn_size)
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
  local grad_context_bwd = ModelUtils.reuseTensor(self.gradContextBwdProto,
                                                  { batch.size, batch.source_length, self.args.rnn_size })

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

return BiEncoder
