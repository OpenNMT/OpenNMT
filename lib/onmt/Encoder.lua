--[[ Encoder is a unidirectional Sequencer used for the source language.

    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     .      .      .             .
     |      |      |             |
    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     |      |      |             |
    x_1    x_2    x_3           x_n


Inherits from [onmt.Sequencer](lib+onmt+Sequencer).
--]]
local Encoder, parent = torch.class('onmt.Encoder', 'onmt.Sequencer')

--[[ Construct an encoder layer. 

Parameters:

  * `args` - global options.
  * `network` - optional recurrent step template.
]]
function Encoder:__init(args, network)
  parent.__init(self, args, network or self:_buildModel(args))

  -- Prototype for preallocated context vector.
  self.contextProto = torch.Tensor()
end

--[[ Build one time-step of an encoder

Parameters:

  * `args` - global args.

Returns: An nn-graph mapping

  $${(c^1_{t-1}, h^1_{t-1}, .., c^L_{t-1}, h^L_{t-1}, x_t) =>
  (c^1_{t}, h^1_{t}, .., c^L_{t}, h^L_{t})}$$

  Where ${c^l}$ and ${h^l}$ are the hidden and cell states at each layer,
  ${x_t}$ is a sparse word to lookup.
--]]
function Encoder:_buildModel(args)
  local inputs = {}
  local states = {}

  -- Inputs are previous layers first.
  for _ = 1, args.num_layers do
    local c0 = nn.Identity()() -- batch_size x rnn_size
    table.insert(inputs, c0)
    table.insert(states, c0)

    local h0 = nn.Identity()() -- batch_size x rnn_size
    table.insert(inputs, h0)
    table.insert(states, h0)
  end

  local x = nn.Identity()() -- batch_size
  table.insert(inputs, x)

  -- Compute word embedding.
  local input = onmt.WordEmbedding(args.vocab_size, args.word_vec_size, args.pre_word_vecs, args.fix_word_vecs)(x)

  table.insert(states, input)

  -- Forward states and input into the RNN.
  local outputs = onmt.LSTM(args.num_layers, args.word_vec_size, args.rnn_size, args.dropout)(states)

  return nn.gModule(inputs, { outputs })
end

--[[Compute the context representation of an input.

Parameters:

  * `batch` - a [batch struct](lib+data/#opennmtdata) as defined data.lua.

Returns:

  1. - final hidden states
  2. - context matrix H
--]]
function Encoder:forward(batch)

  -- TODO: Change `batch` to `input`.
  
  local final_states

  if self.statesProto == nil then
    self.statesProto = utils.Tensor.initTensorTable(self.args.num_layers * 2,
                                                    self.stateProto,
                                                    { batch.size, self.args.rnn_size })
  end

  -- Make initial states c_0, h_0.
  local states = utils.Tensor.reuseTensorTable(self.statesProto, { batch.size, self.args.rnn_size })

  -- Preallocated output matrix.
  local context = utils.Tensor.reuseTensor(self.contextProto,
                                           { batch.size, batch.source_length, self.args.rnn_size })

  if self.args.mask_padding and not batch.source_input_pad_left then
    final_states = utils.Tensor.recursiveClone(states)
  end
  if self.train then
    self.inputs = {}
  end

  -- Act like nn.Sequential and call each clone in a feed-forward
  -- fashion.
  for t = 1, batch.source_length do

    -- Construct "inputs". Prev states come first then source.
    local inputs = {}
    utils.Table.append(inputs, states)
    table.insert(inputs, batch.source_input[t])

    if self.train then
      -- Remember inputs for the backward pass.
      self.inputs[t] = inputs
    end

    states = self:net(t):forward(inputs)

    -- Special case padding.
    if self.args.mask_padding then
      for b = 1, batch.size do
        if batch.source_input_pad_left and t <= batch.source_length - batch.source_size[b] then
          for j = 1, #states do
            states[j][b]:zero()
          end
        elseif not batch.source_input_pad_left and t == batch.source_size[b] then
          for j = 1, #states do
            final_states[j][b]:copy(states[j][b])
          end
        end
      end
    end

    -- Copy output (h^L_t = states[#states]) to context.
    context[{{}, t}]:copy(states[#states])
  end

  if final_states == nil then
    final_states = states
  end

  return final_states, context
end

--[[ Backward pass (only called during training)

Parameters:

  * `batch` - must be same as for forward
  * `grad_states_output` gradient of loss wrt last state
  * `grad_context_output` - gradient of loss wrt full context.

Returns: nil
--]]
function Encoder:backward(batch, grad_states_output, grad_context_output)
  -- TODO: change this to (input, gradOutput) as in nngraph.
  if self.gradOutputsProto == nil then
    self.gradOutputsProto = utils.Tensor.initTensorTable(self.args.num_layers * 2,
                                                         self.gradOutputProto,
                                                         { batch.size, self.args.rnn_size })
  end

  local grad_states_input = utils.Tensor.copyTensorTable(self.gradOutputsProto, grad_states_output)

  for t = batch.source_length, 1, -1 do
    -- Add context gradients to last hidden states gradients.
    grad_states_input[#grad_states_input]:add(grad_context_output[{{}, t}])

    local grad_input = self:net(t):backward(self.inputs[t], grad_states_input)

    -- Prepare next encoder output gradients.
    for i = 1, #grad_states_input do
      grad_states_input[i]:copy(grad_input[i])
    end
  end
end
