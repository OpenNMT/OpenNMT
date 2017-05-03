--[[ Default decoder generator.
     Given RNN state, produce categorical distribution for tokens and features

     Simply implements $$softmax(W h b)$$.

     version 2: merge FeaturesGenerator and Generator - the generator nn is a table
--]]
local Generator, parent = torch.class('onmt.Generator', 'onmt.Network')

-- for back compatibility - still declare FeaturesGenerator - but no need to define it
torch.class('onmt.FeaturesGenerator', 'onmt.Generator')

function Generator.declareOpts(cmd)
  onmt.ISGenerator.declareOpts(cmd)
end

function Generator:__init(opt, sizes)
  parent.__init(self)
  self:_buildGenerator(opt, sizes)
  -- for backward compatibility with previous model
  self.version = 2
end

function Generator:_simpleGeneratorLayer(input, output)
  return nn.Sequential()
                      :add(nn.Linear(input, output))
                      :add(nn.LogSoftMax())
end

function Generator:_buildGenerator(opt, sizes)
  local generator = nn.ConcatTable()
  local rnn_size = opt.rnn_size

  for i = 1, #sizes do
    generator:add(self:_simpleGeneratorLayer(rnn_size, sizes[i]))
  end

  self:set(generator)
end

--[[ If the target vocabulary for the batch is not full vocabulary ]]
function Generator:setTargetVoc(_)
end

--[[ If the target vocabulary for the batch is not full vocabulary ]]
function Generator:unsetTargetVoc(_)
end

--[[ Release Generator for inference only ]]
function Generator:release()
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
  input = type(input) == 'table' and input[1] or input
  self.output = self.net:updateOutput(input)
  return self.output
end

function Generator:updateGradInput(input, gradOutput)
  input = type(input) == 'table' and input[1] or input
  self.gradInput = self.net:updateGradInput(input, gradOutput)
  return self.gradInput
end

function Generator:accGradParameters(input, gradOutput, scale)
  input = type(input) == 'table' and input[1] or input
  self.net:accGradParameters(input, gradOutput, scale)
end
