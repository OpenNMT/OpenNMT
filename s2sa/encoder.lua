local model_utils = require 's2sa.model_utils'
require 's2sa.sequencer'

local Encoder, Sequencer = torch.class('Encoder', 'Sequencer')

function Encoder:__init(args, opt)
  Sequencer.__init(self, args, opt)
end

function Encoder:forward(batch)
  local default_states = model_utils.reset_state(self.init_states, batch.size)

  return Sequencer.forward(self, default_states, batch.source_input)
end

return Encoder
