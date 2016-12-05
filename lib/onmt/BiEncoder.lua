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

Inherits from [onmt.Sequencer](lib+onmt+Sequencer).

--]]
local BiEncoder, parent = torch.class('onmt.BiEncoder', 'nn.Container')

--[[ Creates two Encoder's (encoder.lua) `net_fwd` and `net_bwd`.
  The two are combined use `merge` operation (concat/sum).
]]
function BiEncoder:__init(pretrained, input, rnn, merge)
  parent.__init(self)

  if pretrained then
    self.fwd = onmt.Encoder.new(pretrained.modules[1])
    self.bwd = onmt.Encoder.new(pretrained.modules[2])

    self.merge = pretrained.merge
    self.rnn_size = pretrained.rnn_size
    self.hidden_size = pretrained.hidden_size
    self.num_effective_layers = pretrained.num_effective_layers
  else
    self.fwd = onmt.Encoder.new(nil, input, rnn)
    self.bwd = onmt.Encoder.new(nil, input:clone(), rnn:clone())

    self.merge = merge

    self.rnn_size = rnn.output_size
    self.num_effective_layers = rnn.num_effective_layers

    if self.merge == 'concat' then
      self.hidden_size = self.rnn_size * 2
    else
      self.hidden_size = self.rnn_size
    end
  end

  self:add(self.fwd)
  self:add(self.bwd)

  -- Prototype for preallocated full context vector.
  self.contextProto = torch.Tensor()

  -- Prototype for preallocated full hidden states tensors.
  self.stateProto = torch.Tensor()

  -- Prototype for preallocated gradient of the backward context
  self.gradContextBwdProto = torch.Tensor()
end

--[[ Return data to serialize. ]]
function BiEncoder:serialize()
  return {
    modules = self.modules,
    merge = self.merge,
    rnn_size = self.rnn_size,
    hidden_size = self.hidden_size,
    num_effective_layers = self.num_effective_layers
  }
end

function BiEncoder:maskPadding()
  self.fwd:maskPadding()
  self.bwd:maskPadding()
end

function BiEncoder:forward(batch)
  if self.statesProto == nil then
    self.statesProto = utils.Tensor.initTensorTable(self.num_effective_layers,
                                                    self.stateProto,
                                                    { batch.size, self.hidden_size })

    if self.train then
      self.bwd:shareInput(self.fwd)
    end
  end

  local states = utils.Tensor.reuseTensorTable(self.statesProto, { batch.size, self.hidden_size })
  local context = utils.Tensor.reuseTensor(self.contextProto,
                                           { batch.size, batch.source_length, self.hidden_size })

  local fwd_states, fwd_context = self.fwd:forward(batch)
  reverse_input(batch)
  local bwd_states, bwd_context = self.bwd:forward(batch)
  reverse_input(batch)

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
  local grad_context_bwd = utils.Tensor.reuseTensor(self.gradContextBwdProto,
                                                    { batch.size, batch.source_length, self.rnn_size })

  for t = 1, batch.source_length do
    grad_context_bwd[{{}, t}]:copy(grad_context_output_bwd[{{}, batch.source_length - t + 1}])
  end

  self.bwd:backward(batch, grad_states_output_bwd, grad_context_bwd)
end
