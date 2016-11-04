require 'nn'
require 'string'
require 'nngraph'
require 's2sa.encoder'
require 's2sa.decoder'
require 's2sa.generator'
local constants = require 's2sa.beam.constants'
local State = require 's2sa.beam.state'
local cuda = require 's2sa.cuda'
local Dict = require 's2sa.dict'
local features = require 's2sa.beam.features'
local tokens = require 's2sa.beam.tokens'
local beam_utils = require 's2sa.beam.utils'
local table_utils = require 's2sa.table_utils'

-- globals
local model
local model_opt

local src_dict
local targ_dict
local char_dict

local word2charidx_targ
local init_fwd_enc = {}
local init_fwd_dec = {}

local context_proto
local context_proto2
local decoder_softmax
local decoder_attn
local softmax_layers
local opt = {}


-- Here is a tricky module: it applies the softmax over the input
-- and zero the part of the output which is padded in the input.
-- It works with or without beam.
local function masked_softmax(source_sizes, with_beam)
  local num_sents = #source_sizes
  local input = nn.Identity()()
  local softmax = nn.SoftMax()(input) -- State.iteration*num_sents x State.source_length

  -- now we are masking the part of the output we don't need
  local tab
  if with_beam then
    tab = nn.SplitTable(2)(nn.View(State.iteration, num_sents, State.source_length)(softmax)) -- num_sents x { State.iteration x State.source_length }
  else
    tab = nn.SplitTable(1)(softmax) -- num_sents x { State.source_length }
  end
  local par = nn.ParallelTable()

  for b = 1, num_sents do
    local pad_length = State.source_length - source_sizes[b]
    local dim = 2
    if not with_beam then
      dim = 1
    end

    local seq = nn.Sequential()
    seq:add(nn.Narrow(dim, pad_length + 1, source_sizes[b]))
    seq:add(nn.Padding(1, -pad_length, 1, 0))
    par:add(seq)
  end

  local out_tab = par(tab) -- num_sents x { State.iteration x State.source_length }
  local output = nn.JoinTable(1)(out_tab) -- num_sents*State.iteration x State.source_length
  if with_beam then
    output = nn.View(num_sents, State.iteration, State.source_length)(output)
    output = nn.Transpose({1,2})(output) -- State.iteration x num_sents x State.source_length
    output = nn.View(State.iteration*num_sents, State.source_length)(output)
  else
    output = nn.View(num_sents, State.source_length)(output)
  end

  -- make sure the vector sums to 1 (softmax output)
  output = nn.Normalize(1)(output)

  return nn.gModule({input}, {output})
end

local function replace_attn_softmax(source_sizes, with_beam)
  decoder_attn:replace(function(module)
    if module.name == 'softmax_attn' then
      local mod = masked_softmax(source_sizes, with_beam)
      if cuda.activated then
        mod = mod:cuda()
      elseif opt.float == 1 then
        mod = mod:float()
      end
      mod.name = 'softmax_attn'
      decoder_softmax = mod
      return mod
    else
      return module
    end
  end)
end

local function reset_attn_softmax()
  decoder_attn:replace(function(module)
      if module.name == 'softmax_attn' then
        local mod = nn.SoftMax()
        mod.name = 'softmax_attn'
        if opt.gpuid >= 0 then
          mod = mod:cuda()
        elseif opt.float == 1 then
          mod = mod:float()
        end
        decoder_softmax = mod
        return mod
      else
        return module
      end
  end)
end

local function generate_beam(K, max_sent_l, source, source_features, gold)
  local source_l = beam_utils.get_max_length(source)

  beam_utils.source_length = source_l
  State.iteration = K

  if opt.gpuid > 0 and opt.gpuid2 > 0 then
    cutorch.setDevice(opt.gpuid)
  end

  local batch = #source

  --reset decoder initial states
  local initial = State.initial(constants.START)

  features.reset(batch, K)

  local n = max_sent_l
  -- Backpointer table.
  local prev_ks = torch.LongTensor(batch, n, K):fill(constants.PAD)
  -- Current States.
  local next_ys = torch.LongTensor(batch, n, K):fill(constants.PAD)
  -- Current Scores.
  local scores = torch.FloatTensor(batch, n, K):zero()

  local next_ys_features = features.get_next_features(batch, n, K)

  local attn_weights = {}
  local states = {}
  for b = 1, batch do
    table.insert(attn_weights, {{initial}}) -- store attn weights
    table.insert(states, {{initial}}) -- store predicted word idx
    next_ys[b][1][1] = State.next(initial)
  end

  local source_input = beam_utils.get_input(source, source_l, true, model_opt.use_chars_enc == 1)
  local source_features_input = features.get_source_features_input(source_features, source_l, true)

  local rnn_state_enc = {}

  for i = 1, #init_fwd_enc do
    init_fwd_enc[i]:resize(batch, model_opt.rnn_size)
    table.insert(rnn_state_enc, init_fwd_enc[i]:zero())
  end

  local context = context_proto[{{}, {1,source_l}}]:clone() -- 1 x source_l x rnn_size
  context = context
    :resize(batch, source_l, model_opt.rnn_size)
    :view(1, batch, source_l, model_opt.rnn_size)
    :zero()

  for t = 1, source_l do
    local encoder_input = beam_utils.get_encoder_input(source_input[t],
                                            source_features_input[t],
                                            rnn_state_enc)
    local out = model[1]:forward(encoder_input)

    if batch > 1 then
      beam_utils.ignore_padded_output(t, source, out)
    end

    rnn_state_enc = out
    context[{{}, {}, t}]:copy(out[#out])
  end

  local rnn_state_dec = {}
  for i = 1, #init_fwd_dec do
    init_fwd_dec[i]:resize(K * batch, model_opt.rnn_size)
    table.insert(rnn_state_dec, init_fwd_dec[i]:zero())
  end

  if model_opt.init_dec == 1 then
    for L = 1, model_opt.num_layers do
      rnn_state_dec[L*2-1+model_opt.input_feed]:copy(
        rnn_state_enc[L*2-1]
          :view(1, batch, model_opt.rnn_size)
          :expand(K, batch, model_opt.rnn_size)
          :contiguous()
          :view(K*batch, model_opt.rnn_size))
      rnn_state_dec[L*2+model_opt.input_feed]:copy(
        rnn_state_enc[L*2]
          :view(1, batch, model_opt.rnn_size)
          :expand(K, batch, model_opt.rnn_size)
          :contiguous()
          :view(K*batch, model_opt.rnn_size))
    end
  end

  if model_opt.brnn == 1 then
    local final_rnn_state_enc = {}
    for i = 1, #rnn_state_enc do
      rnn_state_enc[i]:zero()
      table.insert(final_rnn_state_enc, rnn_state_enc[i]:clone())
    end

    for t = source_l, 1, -1 do
      local encoder_input = beam_utils.get_encoder_input(source_input[t],
                                              source_features_input[t],
                                              rnn_state_enc)
      local out = model[4]:forward(encoder_input)

      if batch > 1 then
        beam_utils.ignore_padded_output(t, source, out)
      end

      for b = 1, batch do
        if t == source_l - source[b]:size(1) + 1 then
          for j = 1, #final_rnn_state_enc do
            final_rnn_state_enc[j][b]:copy(out[j][b])
          end
        end
      end

      rnn_state_enc = out
      context[{{}, {}, t}]:add(out[#out])
    end

    if model_opt.init_dec == 1 then
      for L = 1, model_opt.num_layers do
        rnn_state_dec[L*2-1+model_opt.input_feed]:add(
          final_rnn_state_enc[L*2-1]
            :view(1, batch, model_opt.rnn_size)
            :expand(K, batch, model_opt.rnn_size)
            :contiguous()
            :view(K*batch, model_opt.rnn_size))
        rnn_state_dec[L*2+model_opt.input_feed]:add(
          final_rnn_state_enc[L*2]
            :view(1, batch, model_opt.rnn_size)
            :expand(K, batch, model_opt.rnn_size)
            :contiguous()
            :view(K*batch, model_opt.rnn_size))
      end
    end
  end

  if gold ~= nil then gold:init(model_opt, rnn_state_dec) end

  context = context
    :expand(K, batch, source_l, model_opt.rnn_size)
    :contiguous()
    :view(K*batch, source_l, model_opt.rnn_size)

  if opt.gpuid > 0 and opt.gpuid2 > 0 then
    cutorch.setDevice(opt.gpuid2)
    local context2 = context_proto2[{{1, K}, {1, source_l}}]
    context2 = context2
      :resize(K*batch, source_l, model_opt.rnn_size)
      :zero()
    context2:copy(context)
    context = context2
  end

  reset_attn_softmax()

  local out_float = torch.FloatTensor()

  local i = 1
  local done = {}
  local found_eos = {}

  local end_score = {}
  local end_hyp = {}
  local end_attn_weights = {}

  local max_score = {}
  local max_hyp = {}
  local max_k = {}
  local max_attn_weights = {}

  local remaining_sents = batch
  local new_context = context:clone()
  local batch_idx = {}

  for b = 1, batch do
    done[b] = false
    found_eos[b] = false
    max_score[b] = -1e9
    end_score[b] = -1e9

    table.insert(max_k, 1)
    table.insert(batch_idx, b)
  end

  while (remaining_sents > 0) and (i < n) do
    i = i+1

    local decoder_input1
    if model_opt.use_chars_dec == 1 then
      decoder_input1 = torch.LongTensor(K, remaining_sents, model_opt.max_word_l):zero()
    else
      decoder_input1 = torch.LongTensor(K, remaining_sents):zero()
    end

    local decoder_input1_features = features.get_decoder_input(K, remaining_sents)

    local source_sizes = {}

    for b = 1, batch do
      if not done[b] then
        local idx = batch_idx[b]
        table.insert(source_sizes, source[b]:size(1))

        states[b][i] = {}
        attn_weights[b][i] = {}

        if model_opt.use_chars_dec == 1 then
          decoder_input1[{{}, idx}]
            :copy(word2charidx_targ:index(1, next_ys[b]:narrow(1,i-1,1):squeeze()))
        else
          local prev_out = next_ys[b]:narrow(1,i-1,1):squeeze()
          if opt.beam == 1 then
            decoder_input1[{{}, idx}]:copy(torch.LongTensor({prev_out}))
          else
            decoder_input1[{{}, idx}]:copy(prev_out)
          end
          for j = 1, features.num_target_features do
            decoder_input1_features[j][{{}, idx}]:copy(next_ys_features[i-1][j][b])
          end
        end
      end
    end

    if model_opt.use_chars_dec == 1 then
      decoder_input1 = decoder_input1:view(K * remaining_sents, model_opt.max_word_l)
    else
      decoder_input1 = decoder_input1:view(K * remaining_sents)
    end

    for j = 1, features.num_target_features do
      if model_opt.target_features_lookup[j] == true then
        decoder_input1_features[j] = decoder_input1_features[j]:view(K * remaining_sents)
      else
        decoder_input1_features[j] = decoder_input1_features[j]:view(K * remaining_sents, -1)
      end
    end

    local decoder_input = beam_utils.get_decoder_input(decoder_input1,
                                            decoder_input1_features,
                                            rnn_state_dec,
                                            new_context)

    if batch > 1 then
      -- replace the attention softmax with a masked attention softmax
      replace_attn_softmax(source_sizes, true)
    end

    local out_decoder, out = beam_utils.decoder_forward(model, decoder_input)

    if type(out) == "table" then
      for j = 1, #out do
        out[j] = out[j]:view(K, remaining_sents, out[j]:size(2)):transpose(1, 2)
      end
      out_float:resize(out[1]:size()):copy(out[1])
    else
      out = out:view(K, remaining_sents, out:size(2)):transpose(1, 2)
      out_float:resize(out:size()):copy(out)
    end

    beam_utils.update_rnn_state(rnn_state_dec, out_decoder)

    if model_opt.start_symbol == 1 then
      decoder_softmax.output[{{}, 1}]:zero()
      decoder_softmax.output[{{}, source_l}]:zero()
    end

    local softmax_out = decoder_softmax.output:view(K, remaining_sents, -1)
    local new_remaining_sents = remaining_sents

    for b = 1, batch do
      if done[b] == false then
        local idx = batch_idx[b]
        for k = 1, K do
          State.disallow(out_float[idx]:select(1, k))
          out_float[idx][k]:add(scores[b][i-1][k])
        end

        -- All the scores available.

        local flat_out = out_float[idx]:view(-1)
        if i == 2 then
          flat_out = out_float[idx][1] -- all outputs same for first batch
        end

        for k = 1, K do
          while true do
            local score, index = flat_out:max(1)
            score = score[1]
            local prev_k, y_i = table_utils.flat_to_rc(out_float[idx], index[1])
            states[b][i][k] = State.advance(states[b][i-1][prev_k], y_i)
            local diff = true
            for k2 = 1, k-1 do
              if State.same(states[b][i][k2], states[b][i][k]) then
                diff = false
              end
            end

            if i < 2 or diff then
              if model_opt.attn == 1 then
                attn_weights[b][i][k] = State.advance(attn_weights[b][i-1][prev_k], softmax_out[prev_k][idx]:clone())
              end
              prev_ks[b][i][k] = prev_k
              next_ys[b][i][k] = y_i
              scores[b][i][k] = score
              flat_out[index[1]] = -1e9
              break -- move on to next k
            end
            flat_out[index[1]] = -1e9
          end
        end

        for j = 1, #rnn_state_dec do
          local view = rnn_state_dec[j]
            :view(K, remaining_sents, model_opt.rnn_size)
          view[{{}, idx}] = view[{{}, idx}]:index(1, prev_ks[b][i])
        end

        features.calculate_hypotheses(next_ys_features, out, K, i, b, idx)

        end_hyp[b] = states[b][i][1]
        end_score[b] = scores[b][i][1]
        if model_opt.attn == 1 then
          end_attn_weights[b] = attn_weights[b][i][1]
        end
        if end_hyp[b][#end_hyp[b]] == constants.END then
          done[b] = true
          found_eos[b] = true
          new_remaining_sents = new_remaining_sents - 1
          batch_idx[b] = 0
        else
          for k = 1, K do
            local possible_hyp = states[b][i][k]
            if possible_hyp[#possible_hyp] == constants.END then
              found_eos[b] = true
              if scores[b][i][k] > max_score[b] then
                max_hyp[b] = possible_hyp
                max_score[b] = scores[b][i][k]
                max_k[b] = k
                if model_opt.attn == 1 then
                  max_attn_weights[b] = attn_weights[b][i][k]
                end
              end
            end
          end
        end
      end
    end

    if new_remaining_sents > 0 and new_remaining_sents ~= remaining_sents then
      -- update sentence indices within the batch and mark sentences to keep
      local to_keep = {}
      local new_idx = 1
      for b = 1, #batch_idx do
        local idx = batch_idx[b]
        if idx > 0 then
          table.insert(to_keep, idx)
          batch_idx[b] = new_idx
          new_idx = new_idx + 1
        end
      end

      to_keep = torch.LongTensor(to_keep)

      -- update rnn_state and context
      for j = 1, #rnn_state_dec do
        rnn_state_dec[j] = rnn_state_dec[j]
          :view(K, remaining_sents, model_opt.rnn_size)
          :index(2, to_keep)
          :view(K*new_remaining_sents, model_opt.rnn_size)
      end

      new_context = new_context
        :view(K, remaining_sents, source_l, model_opt.rnn_size)
        :index(2, to_keep)
        :view(K*new_remaining_sents, source_l, model_opt.rnn_size)
    end

    remaining_sents = new_remaining_sents
    collectgarbage()
  end

  local states_res = {}
  local scores_res = {}
  local attn_weights_res = {}


  for b = 1, batch do
    if opt.simple == 1 or end_score[b] > max_score[b] or not found_eos[b] then
      max_hyp[b] = end_hyp[b]
      max_score[b] = end_score[b]
      max_attn_weights[b] = end_attn_weights[b]
      max_k[b] = 1
    end

    -- remove unnecessary values from the attention vectors
    for j = 2, #max_attn_weights[b] do
      local size = source[b]:size(1)
      max_attn_weights[b][j] = max_attn_weights[b][j]:narrow(1, source_l-size+1, size)
    end

    table.insert(attn_weights_res, max_attn_weights[b][#max_attn_weights[b]])

    local sent_len = #max_hyp[b]

    table.insert(states_res, states[b][sent_len])
    table.insert(scores_res, scores[b][sent_len])

    features.calculate_max_hypothesis(sent_len, max_k, prev_ks, b)
  end

  if gold ~= nil then
    gold:process(batch, model, context, init_fwd_dec, word2charidx_targ, source, max_score)
  end

  return max_hyp, features.max_hypothesis, max_score, max_attn_weights, states_res, scores_res, attn_weights_res
end

local function get_layer(layer)
  if layer.name ~= nil then
    if layer.name == 'decoder_attn' then
      decoder_attn = layer
    elseif layer.name:sub(1,7) == 'softmax' then
      table.insert(softmax_layers, layer)
    end
  end
end

local function init(args, resources_dir)
  -- parse input params
  opt = args

  opt.model = beam_utils.absolute_path(opt.model, resources_dir)
  opt.src_dict = beam_utils.absolute_path(opt.src_dict, resources_dir)
  opt.targ_dict = beam_utils.absolute_path(opt.targ_dict, resources_dir)
  opt.srctarg_dict = beam_utils.absolute_path(opt.srctarg_dict, resources_dir)
  opt.char_dict = beam_utils.absolute_path(opt.char_dict, resources_dir)
  opt.feature_dict_prefix = beam_utils.absolute_path(opt.feature_dict_prefix, resources_dir)

  cuda.init(opt)

  print('loading ' .. opt.model .. '...')
  local checkpoint = torch.load(opt.model)
  print('done!')

  -- load model and word2idx/idx2word dictionaries
  model, model_opt = checkpoint[1], checkpoint[2]
  for i = 1, #model do
    model[i]:evaluate()
    if opt.float == 1 then
      model[i] = model[i]:float()
    end
  end
  -- for backward compatibility
  model_opt.brnn = model_opt.brnn or 0
  model_opt.input_feed = model_opt.input_feed or 1
  model_opt.attn = model_opt.attn or 1
  model_opt.num_source_features = model_opt.num_source_features or 0
  model_opt.num_target_features = model_opt.num_target_features or 0
  model_opt.guided_alignment = model_opt.guided_alignment or 0

  beam_utils.init(model_opt, opt.max_sent_l, opt.float)

  tokens.init(model_opt.use_chars_enc == 1 or model_opt.use_chars_dec == 1, opt.replace_unk, opt.srctarg_dict)

  if opt.src_dict == "" then
    src_dict = checkpoint[4]
  else
    src_dict = Dict.new(opt.src_dict)
  end

  if opt.targ_dict == "" then
    targ_dict = checkpoint[5]
  else
    targ_dict = Dict.new(opt.targ_dict)
  end

  features.init({
    src_dict = opt.feature_dict_prefix == "" and checkpoint[6] or nil,
    targ_dict = opt.feature_dict_prefix == "" and checkpoint[7] or nil,
    dicts_prefix = opt.feature_dict_prefix,
    num_source_features = model_opt.num_source_features,
    num_target_features = model_opt.num_target_features,
    source_features_lookup = model_opt.source_features_lookup,
    target_features_lookup = model_opt.target_features_lookup
  })

  if opt.char_dict == "" then
    char_dict = checkpoint[8]
  else
    char_dict = Dict.new(opt.char_dict)
  end

  -- load character dictionaries if needed
  if model_opt.use_chars_enc == 1 or model_opt.use_chars_dec == 1 then
    char_dict = Dict.new(opt.char_dict)
    model[1]:apply(get_layer)

    if model_opt.use_chars_dec == 1 then
      word2charidx_targ = torch.LongTensor(#targ_dict.idx_to_label, model_opt.max_word_l):fill(constants.PAD)
      for i = 1, #targ_dict.idx_to_label do
        word2charidx_targ[i] = tokens.words_to_chars_idx(targ_dict:lookup(i), char_dict, model_opt.max_word_l, word2charidx_targ[i])
      end
    end
  end

  if opt.gpuid >= 0 then
    for i = 1, #model do
      if opt.gpuid2 > 0 then
        if i == 1 or i == 4 then
          cutorch.setDevice(opt.gpuid)
        else
          cutorch.setDevice(opt.gpuid2)
        end
      end
      model[i]:double():cuda()
      model[i]:evaluate()
    end
  end

  softmax_layers = {}
  model[2].network:apply(get_layer)
  local attn_layer
  if model_opt.attn == 1 then
    decoder_attn:apply(get_layer)
    decoder_softmax = softmax_layers[1]
    attn_layer = torch.zeros(opt.beam, opt.max_sent_l)
  end

  context_proto = torch.zeros(1, opt.max_sent_l, model_opt.rnn_size)
  local h_init_dec = torch.zeros(opt.beam, model_opt.rnn_size)
  local h_init_enc = torch.zeros(1, model_opt.rnn_size)
  if opt.gpuid >= 0 then
    h_init_enc = h_init_enc:cuda()
    h_init_dec = h_init_dec:cuda()

    if opt.gpuid2 > 0 then
      cutorch.setDevice(opt.gpuid)
      context_proto = context_proto:cuda()
      cutorch.setDevice(opt.gpuid2)
      context_proto2 = torch.zeros(opt.beam, opt.max_sent_l, model_opt.rnn_size):cuda()
    else
      if opt.gpuid > 0 then
        cutorch.setDevice(opt.gpuid)
      end
      context_proto = context_proto:cuda()
    end
    if model_opt.attn == 1 then
      attn_layer:cuda()
    end
  elseif opt.float == 1 then
    context_proto = context_proto:float()
    h_init_dec = h_init_dec:float()
    h_init_enc = h_init_enc:float()
  end

  init_fwd_enc = {}
  init_fwd_dec = {}
  if model_opt.input_feed == 1 then
    table.insert(init_fwd_dec, h_init_dec:clone())
  end

  for _ = 1, model_opt.num_layers do
    table.insert(init_fwd_enc, h_init_enc:clone())
    table.insert(init_fwd_enc, h_init_enc:clone())
    table.insert(init_fwd_dec, h_init_dec:clone()) -- memory cell
    table.insert(init_fwd_dec, h_init_dec:clone()) -- hidden state
  end
end

local function build_target_tokens(sentences)
  local target_batch = {}
  local target_features_batch = {}

  for i = 1, #sentences do
    local target_tokens, target_features_str = features.extract(sentences[i])
    local target = tokens.to_words_idx(target_tokens, targ_dict.label_to_idx, 1)
    local target_features = features.to_target_features_idx(target_features_str, 1, 1)

    table.insert(target_batch, target)
    table.insert(target_features_batch, target_features)
  end

  return target_batch, target_features_batch
end

--[[
  @table token

  @field value - token value (word)
  @field [attention] - attention tensor used for predicting the token
  @field [range] - token range:
    {
      begin = <begin char index>,
      end = <end char index (not including)>
    }
]]

--[[
  @function search
  @brief Performs a beam search.

  @param batch - a batch of sentences to translate.
  Each batch element can be either an array of tokens (from evaluate.lua)
  or a key/value table (from extEngineAPI.lua) which contains:
    {
      options = <array of translation options>,
      tokens = <array of cleaned tokens represents a sentence in batch>
      features = <array of features>
    }
  @param gold - instance of s2sa.beam.gold used to calculate gold scores
  @return result - a key/value table contains:
    {
      cleaned_pred_features_batch - array of predicted feature batch
      cleaned_pred_tokens_batch - array of cleaned predicted batch
      pred_tokens_batch - array of predicted batch
      info_batch - table containing various batch info, each element has following info:
        {
          pred_score = <prediction score>,
          pred_words = <prediction words count>,
          nbests = [
            {
              tokens = <array of tokens>,
              score = <prediction score>
            }
          ]
        }
    }
]]
local function search(batch, gold)
  local source_batch = {}
  local source_str_batch = {}
  local source_features_batch = {}

  for i = 1, #batch do
    local cleaned_tokens, source_features_str
    if batch[i].tokens ~= nil then
      cleaned_tokens = batch[i].tokens
      source_features_str = batch[i].features
    else
      cleaned_tokens, source_features_str = features.extract(batch[i])
    end

    local source, source_str
    local source_features = features.to_source_features_idx(source_features_str, model_opt.start_symbol, 0)
    if model_opt.use_chars_enc == 1 then
      source, source_str = tokens.to_chars_idx(cleaned_tokens, char_dict, model_opt.max_word_l, model_opt.start_symbol)
    else
      source, source_str = tokens.to_words_idx(cleaned_tokens, src_dict.label_to_idx, model_opt.start_symbol)
    end

    table.insert(source_batch, source)
    table.insert(source_str_batch, source_str)
    table.insert(source_features_batch, source_features)
  end

  local pred_batch, pred_features_batch, pred_score_batch, attn_batch, all_sents_batch, all_scores_batch = generate_beam(
    opt.beam, opt.max_sent_l, source_batch, source_features_batch, gold)

  local cleaned_pred_features_batch = {}
  local cleaned_pred_tokens_batch = {}
  local pred_tokens_batch = {}
  local info_batch = {}

  for i = 1, #batch do
    local pred_tokens, cleaned_pred_tokens, cleaned_pred_features = tokens.from_words_idx(pred_batch[i], pred_features_batch[i], targ_dict.idx_to_label, features.targ_features_dicts, source_str_batch[i], attn_batch[i], features.num_target_features)

    local info = {
      nbests = {},
      pred_score = pred_score_batch[i],
      pred_words = #pred_batch[i] - 1
    }

    if opt.n_best > 1 and features.num_target_features == 0 then
      for n = 1, opt.n_best do
        local pred_tokens_n = tokens.from_words_idx(all_sents_batch[i][n], pred_features_batch[i],
                                             targ_dict.idx_to_label, features.targ_features_dicts, source_str_batch[i], attn_batch[i], features.num_target_features)
        table.insert(info.nbests, {
          tokens = pred_tokens_n,
          score = all_scores_batch[i][n]
        })
      end
    end

    table.insert(cleaned_pred_features_batch, cleaned_pred_features)
    table.insert(cleaned_pred_tokens_batch, cleaned_pred_tokens)
    table.insert(pred_tokens_batch, pred_tokens)
    table.insert(info_batch, info)
  end

  return {
    cleaned_pred_features_batch = cleaned_pred_features_batch,
    cleaned_pred_tokens_batch = cleaned_pred_tokens_batch,
    pred_tokens_batch = pred_tokens_batch,
    info_batch = info_batch
  }
end

return {
  init = init,
  search = search,
  replace_attn_softmax = replace_attn_softmax,
  build_target_tokens = build_target_tokens
}
