require 'torch'
local constants = require 's2sa.utils.constants'
local cuda = require 's2sa.utils.cuda'
require 's2sa.model'

local function build_network(vocab_size, rnn_size)
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- decoder output

  local map = nn.Linear(rnn_size, vocab_size)(inputs[1])
  local loglk = cuda.nn.LogSoftMax()(map)

  return nn.gModule(inputs, {loglk})
end

local function build_criterion(vocab_size)
  local w = torch.ones(vocab_size)
  w[constants.PAD] = 0
  local criterion = nn.ClassNLLCriterion(w)
  criterion.sizeAverage = false
  return criterion
end


local Generator, Model = torch.class('Generator', 'Model')

function Generator:__init(args, network)
  Model.__init(self)
  self.network = network or build_network(args.vocab_size, args.rnn_size)

  if args.training then
    self.criterion = build_criterion(args.vocab_size)
  end
end

function Generator:forward_one(decoder_output)
  return self.network:forward(decoder_output)
end

function Generator:compute_loss(batch, decoder_outputs)
  local loss = 0

  for t = 1, batch.target_length do
    local output = self:forward_one(decoder_outputs[t])
    loss = loss + self.criterion:forward(output, batch.target_output[t])
  end

  return loss
end

function Generator:training()
  self.network:training()
end

function Generator:evaluate()
  self.network:evaluate()
end

function Generator:convert(f)
  f(self.network)
  f(self.criterion)
end

return Generator
