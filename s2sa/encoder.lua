local model_utils = require 's2sa.model_utils'
local Sequencer = require 's2sa.sequencer'

local Encoder, parent = torch.class('Encoder', 'Sequencer')

function Encoder:__init(args)
  parent:__init(args)
end

function Encoder:forward(batch)
  self:forget()
  self.inputs = model_utils.reset_state(self.init_states, batch.size)
  table.insert(self.inputs, batch.source_input)

  local outputs = self.network:forward(self.inputs)

  local context = outputs[#outputs]
  table.remove(outputs)

  return outputs, context
end

function Encoder:backward(grad_output)
  self.network:backward(self.inputs, grad_output)

  self.word_vecs.gradWeight[1]:zero()
  if self.fix_word_vecs == 1 then
    self.word_vecs.gradWeight:zero()
  end
end

return Encoder
