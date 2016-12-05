local FeatureGenerator, parent = torch.class('onmt.FeatureGenerator', 'nn.Container')

function FeatureGenerator:__init(rnn_size, output_size, features)
  parent.__init(self)
  self.net = self:_buildGenerator(rnn_size, output_size, features)
  self:add(self.net)
end

function FeatureGenerator:_buildGenerator(rnn_size, output_size, features)
  local generator = nn.ConcatTable()

  -- Add default generator.
  generator:add(nn.Sequential()
                  :add(onmt.Generator(rnn_size, output_size))
                  :add(nn.SelectTable(1)))

  -- Add a generator for each target feature.
  for i = 1, #features do
    generator:add(nn.Sequential()
                    :add(nn.Linear(rnn_size, #features[i]))
                    :add(nn.LogSoftMax()))
  end

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
