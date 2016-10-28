require 'torch'

local hdf5 = require 'hdf5'
local model_utils = require 's2sa.model_utils'
local table_utils = require 's2sa.table_utils'

local Sequencer = torch.class('Sequencer')

function Sequencer:__init(args)
  self.network = args.network
  self.fix_word_vecs = args.fix_word_vecs

  self.sequencers = {}

  self.network:apply(function (layer)
      if layer.name == 'word_vecs' then
        self.word_vecs = layer
      elseif layer.name == 'lstm' then
        table.insert(self.sequencers, layer)
      end
  end)

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
