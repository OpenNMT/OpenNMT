-- [[ Default decoder generator. Given RNN state, produce categorical distribution.

Simply Implements $$softmax(W h + b)$$.
--]]
local Generator, parent = torch.class('onmt.Generator', 'nn.Container')


function Generator:__init(rnnSize, outputSize)
  parent.__init(self)
  self.net = self:_buildGenerator(rnnSize, outputSize)
  self:add(self.net)
end

function Generator:_buildGenerator(rnnSize, outputSize)
  return nn.Sequential()
    :add(nn.Linear(rnnSize, outputSize))
    :add(nn:LogSoftMax())
end

function Generator:updateOutput(input)
  self.output = {self.net:updateOutput(input)}
  return self.output
end

function Generator:updateGradInput(input, gradOutput)
  self.gradInput = self.net:updateGradInput(input, gradOutput[1])
  return self.gradInput
end

function Generator:accGradParameters(input, gradOutput, scale)
  self.net:accGradParameters(input, gradOutput[1], scale)
end
