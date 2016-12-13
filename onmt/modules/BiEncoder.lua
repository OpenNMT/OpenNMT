local function reverse_input(batch)
  batch.source_input, batch.source_input_rev = batch.source_input_rev, batch.source_input
  batch.source_input_features, batch.source_input_rev_features = batch.source_input_rev_features, batch.source_input_features
  batch.source_input_pad_left, batch.source_input_rev_pad_left = batch.source_input_rev_pad_left, batch.source_input_pad_left
end

--[[ BiEncoder is a bidirectional Sequencer used for the source language.


 `net_fwd`

    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     .      .      .             .
     |      |      |             |
    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     |      |      |             |
    x_1    x_2    x_3           x_n

 `net_bwd`

    h_1 <= h_2 <= h_3 <= ... <= h_n
     |      |      |             |
     .      .      .             .
     |      |      |             |
    h_1 <= h_2 <= h_3 <= ... <= h_n
     |      |      |             |
     |      |      |             |
    x_1    x_2    x_3           x_n

Inherits from [onmt.Sequencer](onmt+modules+Sequencer).

--]]
local BiEncoder, parent = torch.class('onmt.BiEncoder', 'nn.Container')

--[[ Creates two Encoder's (encoder.lua) `net_fwd` and `net_bwd`.
  The two are combined use `merge` operation (concat/sum).
]]
function BiEncoder:__init(input, rnn, merge)
  parent.__init(self)

  self.fwd = onmt.Encoder.new(input, rnn)
  self.bwd = onmt.Encoder.new(input:clone(), rnn:clone())

  self.args = {}
  self.args.merge = merge

  self.args.rnn_size = rnn.output_size
  self.args.num_effective_layers = rnn.num_effective_layers

  if self.args.merge == 'concat' then
    self.args.hidden_size = self.args.rnn_size * 2
  else
    self.args.hidden_size = self.args.rnn_size
  end

  self:add(self.fwd)
  self:add(self.bwd)

  self:resetPreallocation()
end

--[[ Return a new BiEncoder using the serialized data `pretrained`. ]]
function BiEncoder.load(pretrained)
  local self = torch.factory('onmt.BiEncoder')()

  parent.__init(self)

  self.fwd = onmt.Encoder.load(pretrained.modules[1])
  self.bwd = onmt.Encoder.load(pretrained.modules[2])
  self.args = pretrained.args

  self:add(self.fwd)
  self:add(self.bwd)

  self:resetPreallocation()

  return self
end

--[[ Return data to serialize. ]]
function BiEncoder:serialize()
  return {
    modules = self.modules,
    args = self.args
  }
end

function BiEncoder:resetPreallocation()
  -- Prototype for preallocated full context vector.
  self.contextProto = torch.Tensor()

  -- Prototype for preallocated full hidden states tensors.
  self.stateProto = torch.Tensor()

  -- Prototype for preallocated gradient of the backward context
  self.gradContextBwdProto = torch.Tensor()
end

function BiEncoder:maskPadding()
  self.fwd:maskPadding()
  self.bwd:maskPadding()
end

function BiEncoder:forward(batch)
  if self.statesProto == nil then
    self.statesProto = onmt.utils.Tensor.initTensorTable(self.args.num_effective_layers,
                                                         self.stateProto,
                                                         { batch.size, self.args.hidden_size })

    if self.train then
      self.bwd:shareInput(self.fwd)
    end
  end

  local states = onmt.utils.Tensor.reuseTensorTable(self.statesProto, { batch.size, self.args.hidden_size })
  local context = onmt.utils.Tensor.reuseTensor(self.contextProto,
                                                { batch.size, batch.source_length, self.args.hidden_size })

  local fwd_states, fwd_context = self.fwd:forward(batch)
  reverse_input(batch)
  local bwd_states, bwd_context = self.bwd:forward(batch)
  reverse_input(batch)

  if self.args.merge == 'concat' then
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
  elseif self.args.merge == 'sum' then
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

  if self.args.merge == 'concat' then
    local grad_context_output_split = grad_context_output:chunk(2, 3)
    grad_context_output_fwd = grad_context_output_split[1]
    grad_context_output_bwd = grad_context_output_split[2]

    for i = 1, #grad_states_output do
      local states_split = grad_states_output[i]:chunk(2, 2)
      table.insert(grad_states_output_fwd, states_split[1])
      table.insert(grad_states_output_bwd, states_split[2])
    end
  elseif self.args.merge == 'sum' then
    grad_context_output_fwd = grad_context_output
    grad_context_output_bwd = grad_context_output

    grad_states_output_fwd = grad_states_output
    grad_states_output_bwd = grad_states_output
  end

  self.fwd:backward(batch, grad_states_output_fwd, grad_context_output_fwd)

  -- reverse gradients of the backward context
  local grad_context_bwd = onmt.utils.Tensor.reuseTensor(self.gradContextBwdProto,
                                                         { batch.size, batch.source_length, self.args.rnn_size })

  for t = 1, batch.source_length do
    grad_context_bwd[{{}, t}]:copy(grad_context_output_bwd[{{}, batch.source_length - t + 1}])
  end

  self.bwd:backward(batch, grad_states_output_bwd, grad_context_bwd)
end
