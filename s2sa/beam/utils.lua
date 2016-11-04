local constants = require 's2sa.beam.constants'
local beam_state = require 's2sa.beam.state'
local table_utils = require 's2sa.table_utils'
local path = require 'pl.path'

local BeamUtils = {
  max_length = 0,
  float = false,
  model_options = {}
}

function BeamUtils.init(model_options, max_length, float)
  BeamUtils.model_options = model_options
  BeamUtils.max_length = max_length
  BeamUtils.float = float
end

function BeamUtils.get_max_length(data)
  local max_length = 0
  for b = 1, #data do
    max_length = math.max(max_length, data[b]:size(1))
  end

  return math.min(max_length, BeamUtils.max_length)
end

function BeamUtils.get_input(data, max_length, pad_left, use_chars)
  local input
  if use_chars then
    input = torch.LongTensor(max_length, #data, BeamUtils.model_options.max_word_l):fill(constants.PAD)
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

function BeamUtils.get_encoder_input(source_in, features_in, rnn_state)
  local encoder_input = {source_in}
  if BeamUtils.model_options.num_source_features > 0 then
    table_utils.append(encoder_input, features_in)
  end
  table_utils.append(encoder_input, rnn_state)
  return encoder_input
end

function BeamUtils.get_decoder_input(target_in, features_in, rnn_state, context)
 local decoder_input = {target_in}
  if BeamUtils.model_options.num_target_features > 0 then
    table_utils.append(decoder_input, features_in)
  end
  if BeamUtils.model_options.attn == 1 then
    table_utils.append(decoder_input, {context})
  else
    table_utils.append(decoder_input, {context[{{}, beam_state.source_length}]})
  end
  table_utils.append(decoder_input, rnn_state)
  return decoder_input
end

function BeamUtils.update_rnn_state(rnn_state, out_decoder)
  local pred_idx = #out_decoder
  if BeamUtils.model_options.guided_alignment == 1 or BeamUtils.model_options.model_trained_with_guided_alignment == 1 then
    pred_idx = #out_decoder - 1
  end
  if BeamUtils.model_options.input_feed == 1 then
    rnn_state[1] = out_decoder[pred_idx]
  end
  for j = 1, pred_idx - 1 do
    rnn_state[j+BeamUtils.model_options.input_feed] = out_decoder[j]
  end
end

function BeamUtils.decoder_forward(model, input)
  local out_decoder = model[2]:forward(input)
  local out_decoder_pred_idx = #out_decoder
  if BeamUtils.model_options.guided_alignment == 1 or BeamUtils.model_options.model_trained_with_guided_alignment == 1 then
    out_decoder_pred_idx = #out_decoder - 1
  end
  local out = model[3]:forward(out_decoder[out_decoder_pred_idx]) -- K x vocab_size
  return out_decoder, out
end

function BeamUtils.ignore_padded_output(t, src, out)
  for b = 1, #src do
    -- ignore output of the padded part
    if t <= beam_state.source_length - src[b]:size(1) then
      for j = 1, #out do
        out[j][b]:zero()
      end
    end
  end
end

function BeamUtils.absolute_path(file_path, resources_dir)
  local function isempty(s)
    return s == nil or s == ''
  end

  if not isempty(resources_dir) and not isempty(file_path) then
    file_path = path.join(resources_dir, file_path)
  end

  if not isempty(file_path) then
    assert(path.exists(file_path), 'Path does not exist: ' .. file_path)
  end

  return file_path
end

return BeamUtils
