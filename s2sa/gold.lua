require 'torch'
local path = require 'pl.path'

local Gold = torch.class("Gold")

function Gold:__init(opt)
  self.gold_sentences = {}
  self.score_gold = opt.score_gold
  self.model_options = opt.model_options
  self.score = 0
  self.dict = opt.dict

  -- load gold labels if it exists
  if opt.targ_file and path.exists(opt.targ_file) then
    print('loading GOLD labels at ' .. opt.targ_file)
    local file = io.open(opt.targ_file, 'r')
    for line in file:lines() do
      table.insert(self.gold_sentences, line)
    end
  else
    self.score_gold = false
  end

  self.rnn_state_dec_gold = {}
end

function Gold:init(rnn_state_dec)
  if self.score_gold == false then
    return
  end

  if self.model_options.init_dec == 1 then
    self.rnn_state_dec_gold = {}
    for i = 1, #rnn_state_dec do
      table.insert(self.rnn_state_dec_gold, rnn_state_dec[i][{{1}}]:clone())
    end
  end
end

function Gold:process(gold, model, context, init_fwd_dec, source_l, word2charidx_targ)
  if self.score_gold == false then
    return
  end

  self.score = 0

  local rnn_state_dec = {}
  if self.model_options.init_dec == 1 then
    rnn_state_dec = self.rnn_state_dec_gold
  else
    for fwd_i = 1, #init_fwd_dec do
      table.insert(rnn_state_dec, init_fwd_dec[fwd_i][{{1}}]:zero())
    end
  end

  local target_l = gold:size(1)
  for t = 2, target_l do
    local decoder_input1
    if self.model_options.use_chars_dec == 1 then
      decoder_input1 = word2charidx_targ:index(1, gold[{{t-1}}])
    else
      decoder_input1 = gold[{{t-1}}]
    end
    local decoder_input
    if self.model_options.attn == 1 then
      decoder_input = {decoder_input1, context[{{1}}], table.unpack(rnn_state_dec)}
    else
      decoder_input = {decoder_input1, context[{{1}, source_l}], table.unpack(rnn_state_dec)}
    end
    local out_decoder = model[2]:forward(decoder_input)
    local out = model[3]:forward(out_decoder[#out_decoder]) -- K x vocab_size
    rnn_state_dec = {} -- to be modified later
    for j = 1, #out_decoder - 1 do
      table.insert(rnn_state_dec, out_decoder[j])
    end
    self.score = self.score + out[1][gold[t]]
  end
end

function Gold:log(sent_id, pred_score)
  if self.gold_sentences ~= nil then
    print('GOLD ' .. sent_id .. ': ' .. self.gold_sentences[sent_id])
    if self.score_gold then
      print(string.format("PRED SCORE: %.4f, GOLD SCORE: %.4f", pred_score, self.score))
    end
  end
end

return Gold
