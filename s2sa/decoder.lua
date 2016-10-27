local model_utils = require 's2sa.model_utils'
local Sequencer = require 's2sa.sequencer'

local Decoder, parent = torch.class('Decoder', 'Sequencer')

function Decoder:__init(args)
  parent:__init(args)
end

function Decoder:forward(batch, encoder_states)
  self:forget()

  self.inputs = model_utils.reset_state(self.init_states, batch.size)
  for i = 1, #encoder_states do
    self.inputs[i]:copy(encoder_states[i])
  end
  table.insert(self.inputs, batch.target_input)

  local outputs = self.network:forward(self.inputs)

  local out = outputs[#outputs]
  table.remove(outputs)

  return outputs, out
end

function Decoder:backward(grad_output)
  local decoder_grad_input = self.network:backward(self.inputs, grad_output)

  self.word_vecs.gradWeight[1]:zero()
  if self.fix_word_vecs == 1 then
    self.word_vecs.gradWeight:zero()
  end

  return decoder_grad_input
end

return Decoder
