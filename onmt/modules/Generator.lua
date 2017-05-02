--[[ Default decoder generator. Given RNN state, produce categorical distribution for tokens and features

     Simply implements $$softmax(W h b)$$.

     version 2: merge FeaturesGenerator and Generator - the generator nn is a table
--]]
local Generator, parent = torch.class('onmt.Generator', 'onmt.Network')

-- for back compatibility - still declare FeaturesGenerator - but no need to define it
torch.class('onmt.FeaturesGenerator', 'onmt.Generator')

function Generator:__init(opt, sizes)
  parent.__init(self)
  self:_buildGenerator(opt, sizes)
  -- for backward compatibility with previous model
  self.version = 2
end

function Generator:_buildGenerator(opt, sizes)
  local generator = nn.ConcatTable()
  local rnn_size = opt.rnn_size

  for i = 1, #sizes do
    local feat_generator
    feat_generator = nn.Sequential()
                        :add(nn.Linear(rnn_size, sizes[i]))
                        :add(nn.LogSoftMax())
    generator:add(feat_generator)
  end

  self:set(generator)
end

function Generator.load(generator)
  if not generator.version then
    if torch.type(generator)=='onmt.Generator' then
      -- convert previous generator
      generator:set(nn.ConcatTable():add(generator.net))
    end
    generator.version = 2
  end
  return generator
end

function Generator:updateOutput(input)
  if type(input) == 'table' then
    input = input[1]
  end
  self.output = self.net:updateOutput(input)
  return self.output
end

function Generator:updateGradInput(input, gradOutput)
  if type(input) == 'table' then
    input = input[1]
  end
  self.gradInput = self.net:updateGradInput(input, gradOutput)
  return self.gradInput
end

function Generator:accGradParameters(input, gradOutput, scale)
  if type(input) == 'table' then
    input = input[1]
  end
  self.net:accGradParameters(input, gradOutput, scale)
end
