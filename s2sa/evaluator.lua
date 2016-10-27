require 'torch'
local table_utils = require 's2sa.table_utils'
local model_utils = require 's2sa.model_utils'

local Evaluator = torch.class("Evaluator")

function Evaluator:__init(layers_nb)
  self.layers_nb = layers_nb
end

function Evaluator:process(states, data)
  states.encoder:evaluate()
  states.decoder:evaluate()
  states.attention:evaluate()
  states.generator:evaluate()

  local nll = 0
  local total = 0
  for i = 1, #data do
    local batch = data:get_batch(i)

    local encoder_states, context = states.encoder:forward(batch)
    local _, decoder_out = states.decoder:forward(batch, encoder_states)

    local loss = 0
    for t = 1, batch.target_length do
      local out = decoder_out:select(2, t)

      local attention_output = states.attention:forward({out, context})
      local generator_output = states.generator:forward(out)

      loss = loss + states.criterion:forward(generator_output, batch.target_output[{{}, t}])
    end
    nll = nll + loss
    total = total + batch.target_non_zeros
  end
  local valid = math.exp(nll / total)
  print("Valid", valid)
  collectgarbage()
  return valid
end

return Evaluator
