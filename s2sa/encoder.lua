require 'torch'
local hdf5 = require 'hdf5'
local model_utils = require 's2sa.model_utils'
local table_utils = require 's2sa.table_utils'

local Encoder = torch.class("Encoder")

function Encoder:__init(args)
  self.word_vecs_enc = args.word_vecs_enc
  self.init_fwd_enc = {}
  self.init_bwd_enc = {}
  self.network = args.network

  if args.pre_word_vecs_enc:len() > 0 then
    local f = hdf5.open(args.pre_word_vecs_enc)
    local pre_word_vecs = f:read('word_vecs'):all()
    for i = 1, pre_word_vecs:size(1) do
      self.word_vecs_enc.weight[i]:copy(pre_word_vecs[i])
    end
  end

  self.word_vecs_enc.weight[1]:zero()

  self.encoder_clones = model_utils.clone_many_times(self.network, args.network_size)

  local h_init = torch.zeros(args.max_batch_size, args.rnn_size)
  if args.cuda then
    h_init = h_init:cuda()
  end

  for _ = 1, args.layers_nb do
    table.insert(self.init_fwd_enc, h_init:clone())
    table.insert(self.init_fwd_enc, h_init:clone())
    table.insert(self.init_bwd_enc, h_init:clone())
    table.insert(self.init_bwd_enc, h_init:clone())
  end
end

function Encoder:forward(batch, context)
  local rnn_state_enc = model_utils.reset_state(self.init_fwd_enc, batch.size, 0)

  for t = 1, batch.source_length do
    self.encoder_clones[t]:training()
    local encoder_input = {batch.source_input[t]}
    table_utils.append(encoder_input, rnn_state_enc[t-1])
    local out = self.encoder_clones[t]:forward(encoder_input)
    rnn_state_enc[t] = out
    context[{{},t}]:copy(out[#out])
  end

  return rnn_state_enc
end

return Encoder
