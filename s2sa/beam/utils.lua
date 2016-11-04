local constants = require 's2sa.beam.constants'
local table_utils = require 's2sa.table_utils'

local Common = {
  nn = nn,
  activated = false
}

function Common.init(model_options, max_length, float)
  Common.model_options = model_options
  Common.max_length = max_length
  Common.float = float
  Common.source_length = 0
end

function Common.get_max_length(data)
  local max_length = 0
  for b = 1, #data do
    max_length = math.max(max_length, data[b]:size(1))
  end

  return math.min(max_length, Common.max_length)
end

function Common.get_input(data, max_length, pad_left, use_chars)
  local input
  if use_chars then
    input = torch.LongTensor(max_length, #data, Common.model_options.max_word_l):fill(constants.PAD)
  else
    input = torch.LongTensor(max_length, #data):fill(constants.PAD)
  end
  for b = 1, #data do
    local size = data[b]:size(1)
    if pad_left then
      input[{{}, b}]:narrow(1, max_length-size+1, size):copy(data[b])
    else
      input[{{}, b}]:narrow(1, 1, size):copy(data[b])
    end
  end
  return input
end

function Common.get_features_input(features, use_lookup, max_len, pad_left)
  local batch = #features
  local features_input = {}

  for i = 1, max_len do
    table.insert(features_input, {})
    for j = 1, #features[1][1] do
      local t
      if use_lookup[j] == true then
        t = torch.LongTensor(batch):fill(constants.PAD)
      else
        t = torch.Tensor(batch, features[1][1][j]:size(2)):zero()
        t[{{}, constants.PAD}]:fill(1)
        if Common.float == 1 then
          t = t:float()
        end
      end
      table.insert(features_input[i], t)
    end
  end

  for b = 1, batch do
    for i = 1, #features[b] do
      for j = 1, #features[b][i] do
        local idx
        if pad_left then
          idx = max_len-#features[b]+i
        else
          idx = i
        end
        if use_lookup[j] == true then
          features_input[idx][j][b] = features[b][i][j]
        else
          features_input[idx][j][b]:copy(features[b][i][j])
        end
      end
    end
  end

  return features_input
end

function Common.get_encoder_input(source_in, features_in, rnn_state)
  local encoder_input = {source_in}
  if Common.model_options.num_source_features > 0 then
    table_utils.append(encoder_input, features_in)
  end
  table_utils.append(encoder_input, rnn_state)
  return encoder_input
end

function Common.get_decoder_input(target_in, features_in, rnn_state, context)
 local decoder_input = {target_in}
  if Common.model_options.num_target_features > 0 then
    table_utils.append(decoder_input, features_in)
  end
  if Common.model_options.attn == 1 then
    table_utils.append(decoder_input, {context})
  else
    table_utils.append(decoder_input, {context[{{}, Common.source_length}]})
  end
  table_utils.append(decoder_input, rnn_state)
  return decoder_input
end

function Common.update_rnn_state(rnn_state, out_decoder)
  local pred_idx = #out_decoder
  if Common.model_options.guided_alignment == 1 or Common.model_options.model_trained_with_guided_alignment == 1 then
    pred_idx = #out_decoder - 1
  end
  if Common.model_options.input_feed == 1 then
    rnn_state[1] = out_decoder[pred_idx]
  end
  for j = 1, pred_idx - 1 do
    rnn_state[j+Common.model_options.input_feed] = out_decoder[j]
  end
end

function Common.decoder_forward(model, input)
  local out_decoder = model[2]:forward(input)
  local out_decoder_pred_idx = #out_decoder
  if Common.model_options.guided_alignment == 1 or Common.model_options.model_trained_with_guided_alignment == 1 then
    out_decoder_pred_idx = #out_decoder - 1
  end
  local out = model[3]:forward(out_decoder[out_decoder_pred_idx]) -- K x vocab_size
  return out_decoder, out
end

function Common.ignore_padded_output(t, src, out)
  for b = 1, #src do
    -- ignore output of the padded part
    if t <= Common.source_length - src[b]:size(1) then
      for j = 1, #out do
        out[j][b]:zero()
      end
    end
  end
end

return Common
