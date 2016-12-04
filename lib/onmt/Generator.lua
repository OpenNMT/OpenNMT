local Generator, parent = torch.class('onmt.Generator', 'nn.Container')
require 'nngraph'
function Generator:__init(rnn_size, output_size)
  parent.__init(self)
  self.net = self:_buildGenerator(rnn_size, output_size)
  self:add(self.net)
end

function Generator:_buildGenerator(rnn_size, output_size)
  local input = nn.Identity()()
  local output = nn.LogSoftMax()(nn.Linear(rnn_size, output_size)(input))
  return nn.gModule({input}, {output})
end

function Generator:updateOutput(input)
  self.output = {self.net:updateOutput(input)}
  return self.output
end

function Generator:updateGradInput(input, gradOutput) 
  return self.net:updateGradInput(input, gradOutput[1])
end

function Generator:accGradParameters(input, gradOutput, scale)
  self.net:accGradParameters(input, gradOutput[1], scale)
end
return Generator
