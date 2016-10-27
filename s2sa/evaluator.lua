require 'torch'
local table_utils = require 's2sa.table_utils'
local model_utils = require 's2sa.model_utils'

local Evaluator = torch.class("Evaluator")

function Evaluator:__init(layers_nb)
  self.layers_nb = layers_nb
end

function Evaluator:process(states, data)
  states.encoder:evaluate()
  states.decoder:evaluate() -- just need one clone
  states.generator:evaluate()

  local nll = 0
  local total = 0
  for i = 1, #data do
    local batch = data:get_batch(i)

    local rnn_state_enc = model_utils.reset_state(states.init_fwd_enc, batch.size)
    local context = states.context_proto[{{1, batch.size}, {1, batch.source_length}}]

    -- forward prop encoder
    for t = 1, batch.source_length do
      local encoder_input = {batch.source_input[t]}
      table_utils.append(encoder_input, rnn_state_enc)
      local out = states.encoder:forward(encoder_input)
      rnn_state_enc = out
      context[{{},t}]:copy(out[#out])
    end

    local rnn_state_dec = model_utils.reset_state(states.init_fwd_dec, batch.size)
    for L = 1, self.layers_nb do
      rnn_state_dec[L*2-1]:copy(rnn_state_enc[L*2-1])
      rnn_state_dec[L*2]:copy(rnn_state_enc[L*2])
    end

    local loss = 0
    for t = 1, batch.target_length do
      local decoder_input = {batch.target_input[t], context, table.unpack(rnn_state_dec)}
      local out = states.decoder:forward(decoder_input)

      rnn_state_dec = {}
      for j = 1, #out-1 do
        table.insert(rnn_state_dec, out[j])
      end
      local pred = states.generator:forward(out[#out])

      local input = pred
      local output = batch.target_output[t]

      loss = loss + states.criterion:forward(input, output)
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
