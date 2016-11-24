local constants = require 'lib.utils.constants'
local cuda = require 'lib.utils.cuda'

--[[ Generator manages both the final post-LSTM linear/softmax
 layer and the criterion. Currently it is just a holder used
 by the decoder.


    y_1    y_2    y_3           y_n
     |      |      |             |
     .      .      .             .

--]]
local Generator, parent = torch.class('Generator', 'nn.Module')

local function build_network(vocab_size, rnn_size)
  -- Builds a layer for predicting words.
  -- Layer used maps from (h^L) => (V).
  local inputs = {}
  table.insert(inputs, nn.Identity()())

  local map = nn.Linear(rnn_size, vocab_size)(inputs[1])

  -- Use cudnn logsoftmax if available.
  local loglk = cuda.nn.LogSoftMax()(map)

  return nn.gModule(inputs, {loglk})
end

local function build_criterion(vocab_size)
  -- Build a NLL criterion that ignores padding.
  local w = torch.ones(vocab_size)
  w[constants.PAD] = 0
  local criterion = nn.ClassNLLCriterion(w)
  criterion.sizeAverage = false
  return criterion
end

function Generator:__init(args, network)
  parent.__init(self)
  self.network = network or build_network(args.vocab_size, args.rnn_size)

  if args.training then
    self.criterion = build_criterion(args.vocab_size)
  end
end

function Generator:forward_one(input)
  return self.network:forward(input)
end

function Generator:training()
  self.network:training()
end

function Generator:evaluate()
  self.network:evaluate()
end

return Generator
