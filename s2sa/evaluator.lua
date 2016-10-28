require 'torch'

local Evaluator = torch.class("Evaluator")

function Evaluator:__init()
end

function Evaluator:process(states, data)
  states.encoder:evaluate()
  states.decoder:evaluate()
  states.generator:evaluate()

  local nll = 0
  local total = 0

  for i = 1, #data do
    states.encoder:forget()
    states.decoder:forget()

    local batch = data:get_batch(i)

    local encoder_states, context = states.encoder:forward(batch)
    local _, decoder_out = states.decoder:forward(batch, encoder_states)

    local loss = 0
    for t = 1, batch.target_length do
      local out = decoder_out:select(2, t)
      local generator_output = states.generator.network:forward({out, context})
      loss = loss + states.generator.criterion:forward(generator_output, batch.target_output[{{}, t}])
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
