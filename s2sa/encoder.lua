local model_utils = require 's2sa.model_utils'
require 's2sa.sequencer'

local Encoder, Sequencer = torch.class('Encoder', 'Sequencer')

function Encoder:__init(args, opt)
  Sequencer:__init(args, opt)
end

function Encoder:forward(batch)
  local default_states = model_utils.reset_state(self.init_states, batch.size)
  return Sequencer:forward(default_states, batch.source_input)
end

return Encoder
