require 'torch'
local hdf5 = require 'hdf5'
local model_utils = require 's2sa.model_utils'

local Decoder = torch.class("Decoder")

function Decoder:__init(args)
  self.word_vecs_dec = args.word_vecs_dec
  self.init_fwd_dec = {}
  self.init_bwd_dec = {}
  self.network = args.network
  self.layers_nb = args.layers_nb

  if args.pre_word_vecs_dec:len() > 0 then
    local f = hdf5.open(args.pre_word_vecs_dec)
    local pre_word_vecs = f:read('word_vecs'):all()
    for i = 1, pre_word_vecs:size(1) do
      self.word_vecs_dec.weight[i]:copy(pre_word_vecs[i])
    end
  end

  self.word_vecs_dec.weight[1]:zero()

  self.decoder_clones = model_utils.clone_many_times(self.network, args.network_size)

  local h_init = torch.zeros(args.max_batch_size, args.rnn_size)
  if args.cuda then
    h_init = h_init:cuda()
  end

  table.insert(self.init_bwd_dec, h_init:clone())
  for _ = 1, args.layers_nb do
    table.insert(self.init_fwd_dec, h_init:clone())
    table.insert(self.init_fwd_dec, h_init:clone())
    table.insert(self.init_bwd_dec, h_init:clone())
    table.insert(self.init_bwd_dec, h_init:clone())
  end
end

function Decoder:forward(batch, context, rnn_state_enc)
  local rnn_state_dec = model_utils.reset_state(self.init_fwd_dec, batch.size, 0)

  -- copy encoder last hidden state to decoder initial state
  for L = 1, self.layers_nb do
    rnn_state_dec[0][L*2-1]:copy(rnn_state_enc[batch.source_length][L*2-1])
    rnn_state_dec[0][L*2]:copy(rnn_state_enc[batch.source_length][L*2])
  end

  local preds = {}

  for t = 1, batch.target_length do
    self.decoder_clones[t]:training()
    local decoder_input = {batch.target_input[t], context, table.unpack(rnn_state_dec[t-1])}
    local out = self.decoder_clones[t]:forward(decoder_input)
    local next_state = {}
    table.insert(preds, out[#out])
    for j = 1, #out-1 do
      table.insert(next_state, out[j])
    end
    rnn_state_dec[t] = next_state
  end

  return rnn_state_dec, preds
end

return Decoder
