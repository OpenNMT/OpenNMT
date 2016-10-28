require 'torch'

local hdf5 = require 'hdf5'
local model_utils = require 's2sa.model_utils'
local table_utils = require 's2sa.table_utils'

local Sequencer = torch.class('Sequencer')

function Sequencer:__init(args)
  self.network = self:build_network(args)
  self.fix_word_vecs = args.fix_word_vecs

  if args.pre_word_vecs:len() > 0 then
    local f = hdf5.open(args.pre_word_vecs)
    local pre_word_vecs = f:read('word_vecs'):all()
    for i = 1, pre_word_vecs:size(1) do
      self.word_vecs.weight[i]:copy(pre_word_vecs[i])
    end
  end

  self.word_vecs.weight[1]:zero()

  self.init_states = {}
  for _ = 1, args.num_layers do
    table.insert(self.init_states, args.h_init:clone())
  end
end

function Sequencer:build_network(args)
  self.sequencers = {}

  local inputs = {}
  for l = 1, args.num_layers do
    table.insert(inputs, nn.Identity()()) -- h0: batch_size x rnn_size
  end
  table.insert(inputs, nn.Identity()()) -- x: batch_size x timesteps

  local hidden_states
  local outputs = {}

  for L = 1, args.num_layers do
    local h0 = inputs[L]
    local x
    local input_size

    if L == 1 then
      input_size = args.word_vec_size
      self.word_vecs = nn.LookupTable(args.vocab_size, input_size)
      x = self.word_vecs(inputs[args.num_layers + 1]) -- batch_size x timesteps x word_vec_size
    else
      input_size = args.rnn_size
      x = nn.Dropout(args.dropout, nil, false)(hidden_states) -- batch_size x timesteps x rnn_size
    end

    local rnn = nn.LSTM(input_size, args.rnn_size)
    table.insert(self.sequencers, rnn)
    hidden_states = rnn({h0, x}) -- batch_size x timesteps x rnn_size

    local out = nn.Select(2, -1)(hidden_states) -- last hidden state: batch_size x rnn_size
    table.insert(outputs, out)
  end

  table.insert(outputs, hidden_states) -- a.k.a context for the encoder

  return nn.gModule(inputs, outputs)
end

function Sequencer:forget()
  for i = 1, #self.sequencers do
    self.sequencers[i]:resetStates()
  end
end

function Sequencer:training()
  self.network:training()
end

function Sequencer:evaluate()
  self.network:evaluate()
end

return Sequencer
