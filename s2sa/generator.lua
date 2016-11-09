require 'torch'
local cuda = require 's2sa.utils.cuda'

local function build_network(vocab_size, rnn_size)
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- decoder output

  local map = nn.Linear(rnn_size, vocab_size)(inputs[1])
  local loglk = cuda.nn.LogSoftMax()(map)

  return nn.gModule(inputs, {loglk})
end

local function build_criterion(vocab_size)
  local w = torch.ones(vocab_size)
  w[1] = 0
  local criterion = nn.ClassNLLCriterion(w)
  criterion.sizeAverage = false
  return criterion
end


local Generator = torch.class('Generator')

function Generator:__init(args, network)
  self.network = network or cuda.convert(build_network(args.vocab_size, args.rnn_size))
  self.criterion = cuda.convert(build_criterion(args.vocab_size))
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

function Generator:forward_backward(batch, decoder_outputs)
  local grad_input = {}

  local loss = 0

  for t = 1, batch.target_length do
    local output = self:forward_one(decoder_outputs[t])

    loss = loss + self.criterion:forward(output, batch.target_output[t]) / batch.size
    local criterion_grad_input = self.criterion:backward(output, batch.target_output[t]) / batch.size

    table.insert(grad_input, self.network:backward(decoder_outputs[t], criterion_grad_input))
  end

  return grad_input, loss
end

function Generator:training()
  self.network:training()
end

function Generator:evaluate()
  self.network:evaluate()
end

function Generator:float()
  self.network:float()
end

function Generator:double()
  self.network:double()
end

function Generator:cuda()
  self.network:cuda()
  self.criterion:cuda()
end

return Generator
