local ModelUtils = require 'lib.utils.model_utils'
local table_utils = require 'lib.utils.table_utils'
require 'lib.sequencer'

--[[ Encoder is a unidirectional Sequencer used for the source language.

    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     .      .      .             .
     |      |      |             |
    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     |      |      |             |
    x_1    x_2    x_3           x_n


Inherits from [Sequencer](lib+sequencer).
--]]
local Encoder, Sequencer = torch.class('Encoder', 'Sequencer')

--[[ Constructor takes global `args` and optional `network`. ]]
function Encoder:__init(args, network)
  Sequencer.__init(self, 'enc', args, network)
  self.mask_padding = args.mask_padding or false

  -- Prototype for preallocated context vector.
  self.contextProto = torch.Tensor()
end

--[[Compute the context representation of an input.

Parameters:

  * `batch` - a [batch struct](lib+data/#opennmtdata) as defined data.lua.

Returns:

  1. - last hidden states
  2. - context matrix H

TODO:

  * Change `batch` to `input`.
--]]
function Encoder:forward(batch)

  local final_states

  if self.statesProto == nil then
    self.statesProto = ModelUtils.initTensorTable(self.args.num_layers * 2,
                                                  self.stateProto,
                                                  { batch.size, self.args.rnn_size })
  end

  -- Make initial states c_0, h_0.
  local states = ModelUtils.reuseTensorTable(self.statesProto, { batch.size, self.args.rnn_size })

  -- Preallocated output matrix.
  local context = ModelUtils.reuseTensor(self.contextProto,
                                         { batch.size, batch.source_length, self.args.rnn_size })

  if self.mask_padding and not batch.source_input_pad_left then
    final_states = table_utils.clone(states)
  end
  if self.train then
    self.inputs = {}
  end

  -- Act like nn.Sequential and call each clone in a feed-forward
  -- fashion.
  for t = 1, batch.source_length do

    -- Construct "inputs". Prev states come first then source.
    local inputs = {}
    table_utils.append(inputs, states)
    table.insert(inputs, batch.source_input[t])

    if self.train then
      -- Remember inputs for the backward pass.
      self.inputs[t] = inputs
    end

    -- TODO: Shouldn't this just be self:net?
    states = Sequencer.net(self, t):forward(inputs)


    -- Special case padding.
    if self.mask_padding then
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
  * `grad_states_output`
  * `grad_context_output` - gradient of loss
      wrt last states and context.

TODO: change this to (input, gradOutput) as in nngraph.
--]]
function Encoder:backward(batch, grad_states_output, grad_context_output)
  if self.gradOutputsProto == nil then
    self.gradOutputsProto = ModelUtils.initTensorTable(self.args.num_layers * 2,
                                                       self.gradOutputProto,
                                                       { batch.size, self.args.rnn_size })
  end

  local grad_states_input = ModelUtils.copyTensorTable(self.gradOutputsProto, grad_states_output)

  for t = batch.source_length, 1, -1 do
    -- Add context gradients to last hidden states gradients.
    grad_states_input[#grad_states_input]:add(grad_context_output[{{}, t}])

    local grad_input = Sequencer.net(self, t):backward(self.inputs[t], grad_states_input)

    -- Prepare next encoder output gradients.
    for i = 1, #grad_states_input do
      grad_states_input[i]:copy(grad_input[i])
    end
  end
end

return Encoder
