--[[ Default decoder generator.
     Given RNN state, produce categorical distribution for tokens and features

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
    local linear = nn.Linear(rnn_size, sizes[i])
    if i == 1 then
      self.rindexLinear = linear
    end
    generator:add(nn.Sequential()
                    :add(linear)
                    :add(nn.LogSoftMax()))
  end

  self:set(generator)
end

--[[ If the target vocabulary for the batch is not full vocabulary ]]
function Generator:setGeneratorVocab(t)
  self.rindexLinear:RIndex_setOutputIndices(t)
end

--[[ Release Generator for inference only ]]
function Generator:release()
  if self.rindexLinear then
    self.rindexLinear:RIndex_clean()
  end
end

function Generator.load(generator)
  -- Ensure backward compatibility with previous generators.

  if not generator.version then
    if torch.typename(generator) == 'onmt.Generator' then
      generator:set(nn.ConcatTable():add(generator.net))
    elseif torch.typename(generator) == 'onmt.FeaturesGenerator' then
      if torch.typename(generator.net.modules[1]:get(1)) == 'onmt.Generator' then
        generator.net.modules[1] = generator.net.modules[1]:get(1).net
      end
    end
    generator.version = 2
  end

  if not generator.rindexLinear then
    generator:apply(function(m)
      if torch.typename(m) == 'nn.Linear' then
        if not generator.rindexLinear then
          generator.rindexLinear = m
        end
      end
    end)
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
