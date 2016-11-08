require 'torch'
local beam = require 's2sa.beam.main'
local beam_utils = require 's2sa.beam.utils'
local features = require 's2sa.beam.features'
local tokens = require 's2sa.beam.tokens'
local path = require 'pl.path'

local Gold = torch.class("Gold")

function Gold:__init(opt)
  self.sentences = {}
  self.score_gold = opt.score_gold
  self.batch_id = 0
  self.sent_id = 0
  self.gold_score_total = 0
  self.gold_words_total = 0

  if self.score_gold and path.exists(opt.gold_file) then
    print('loading GNEW labels at ' .. opt.gold_file)

    local gold_file = io.open(opt.gold_file, 'r')
    local targ_sents = {}
    while true do
      local line = gold_file:read("*line")
      if line ~= nil then
        table.insert(targ_sents, tokens.from_sentence(line))
      end
      if line == nil or #targ_sents == opt.batch_size then
        table.insert(self.sentences, targ_sents)
        targ_sents = {}
        collectgarbage()
        if line == nil then
          break
        end
      end
    end
  else
    self.score_gold = false
  end

  self.rnn_state_dec_gold = {}
end

function Gold:init(model_options, rnn_state_dec)
  if self.score_gold == false then
    return
  end

  self.model_options = model_options

  if self.model_options.init_dec == 1 then
    self.rnn_state_dec_gold = {}
    for i = 1, #rnn_state_dec do
      table.insert(self.rnn_state_dec_gold, rnn_state_dec[i][{{1}}]:clone())
    end
  end
end

function Gold:process(batch_size, model, context, init_fwd_dec, word2charidx_targ, source, pred_score)
  if self.score_gold == false then
    return
  end

  if self.batch_id > #self.sentences then
    return
  end

  local gold_score = {}

  local sentences = self.sentences[self.batch_id]
  local gold, gold_features = beam.build_target_tokens(sentences)

  for _ = 1, batch_size do
      table.insert(gold_score, 0)
  end

  local rnn_state_dec = {}

  for fwd_i = 1, #init_fwd_dec do
    table.insert(rnn_state_dec, init_fwd_dec[fwd_i]:narrow(1, 1, batch_size):zero())
  end
  if self.model_options.init_dec == 1 then
    for j = 1, #rnn_state_dec do
      rnn_state_dec[j] = self.rnn_state_dec_gold[j]:narrow(1, 1, batch_size)
    end
  end
  local target_l = beam_utils.get_max_length(gold)
  local gold_input = beam_utils.get_input(gold, target_l, false, false)
  local gold_features_input = features.get_target_features_input(gold_features, target_l, false)

  local source_sizes = {}
  for b = 1, batch_size do
    table.insert(source_sizes, source[b]:size(1))
  end
  for t = 2, target_l do
    local decoder_input1
    if self.model_options.use_chars_dec == 1 then
      decoder_input1 = word2charidx_targ:index(1, gold_input[t-1])
    else
      decoder_input1 = gold_input[t-1]
    end
    local decoder_input = beam_utils.get_decoder_input(decoder_input1,
                                            gold_features_input[t-1],
                                            rnn_state_dec,
                                            context:narrow(1, 1, batch_size))
    if batch_size > 1 then
      beam.replace_attn_softmax(source_sizes, false)
    end
    local out_decoder, out = beam_utils.decoder_forward(model, decoder_input)
    beam_utils.update_rnn_state(rnn_state_dec, out_decoder)

    for b = 1, batch_size do
      if t <= gold[b]:size(1) then
        if type(out) == "table" then
          gold_score[b] = gold_score[b] + out[1][b][gold_input[t][b]]
        else
          gold_score[b] = gold_score[b] + out[b][gold_input[t][b]]
        end
      end
    end
  end

  for b = 1, batch_size do
    self.sent_id = self.sent_id + 1

    print('GNEW ' .. self.sent_id .. ': ' .. tokens.to_sentence(gold[b]))
    print(string.format("PRED SCORE: %.4f, GNEW SCORE: %.4f", pred_score[b], gold_score[b]))
    self.gold_score_total = self.gold_score_total + gold_score[b]
    self.gold_words_total = self.gold_words_total + gold[b]:size(1) - 1
  end
end

function Gold:log_results()
  if self.score_gold then
    print(string.format("GNEW AVG SCORE: %.4f, GNEW PPL: %.4f",
      self.gold_score_total / self.gold_words_total,
      math.exp(-self.gold_score_total/self.gold_words_total)))
  end
end

return Gold
