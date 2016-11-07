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
    local batch = data:get_batch(i)

    local encoder_states, context = states.encoder:forward(batch)
    local decoder_outputs = states.decoder:forward(batch, encoder_states, context)

    local loss = states.generator:compute_loss(batch, decoder_outputs)

    nll = nll + loss
    total = total + batch.target_non_zeros
  end

  local valid = math.exp(nll / total)
  print("Valid", valid)
  collectgarbage()
  return valid
end

return Evaluator
