local FeatureGenerator, parent = torch.class('onmt.FeatureGenerator', 'nn.Container')

function FeatureGenerator:__init(rnn_size, output_size, features)
  parent.__init(self)
  self.net = self:_buildGenerator(rnn_size, output_size, features)
  self:add(self.net)
end

--[[ Build the default generator. --]]
function FeatureGenerator:_buildGenerator(rnn_size, output_size, features)
  local split = nn.ConcatTable()
  split:add(nn.Linear(rnn_size, output_size))

  for i = 1, #features do
    split:add(nn.Linear(rnn_size, #features[i]))
  end

  local softmax = nn.ParallelTable()
  softmax:add(nn.LogSoftMax())

  for _ = 1, #features do
    softmax:add(nn.LogSoftMax())
  end

  local generator = nn.Sequential()
    :add(split)
    :add(softmax)

  return generator
end

function FeatureGenerator:updateOutput(input)
  self.output = self.net:updateOutput(input)
  return self.output
end

function FeatureGenerator:updateGradInput(input, gradOutput)
  self.gradInput = self.net:updateGradInput(input, gradOutput)
  return self.gradInput
end

function FeatureGenerator:accGradParameters(input, gradOutput, scale)
  self.net:accGradParameters(input, gradOutput, scale)
end
