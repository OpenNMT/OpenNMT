--[[ Default decoder generator.
     Given RNN state, produce categorical distribution for tokens and features

     Simply implements $$softmax(W h b)$$.

     version 2: merge FeaturesGenerator and Generator - the generator nn is a table
--]]
local Generator, parent = torch.class('onmt.Generator', 'onmt.Network')

-- for back compatibility - still declare FeaturesGenerator - but no need to define it
torch.class('onmt.FeaturesGenerator', 'onmt.Generator')

local options = {
  {
    '-importance_sampling', false,
    [[Use importance sampling approach as approximation of full softmax, target vocabulary is built using sampling.]],
    {
      depends = function(opt)
                  if opt.importance_sampling then
                    if opt.model_type and opt.model_type ~= 'seq2seq' then return false, "only works for seq2seq models." end
                    if opt.sample == 0 then return false, "requires '-sample' option" end
                  end
                  return true
                end
    }
  }
}

function Generator.declareOpts(cmd)
  cmd:setCmdLineOptions(options, 'Generator')
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
    if i == 1 then
      self.rindexLinear = nn.Linear(rnn_size, sizes[i])
      generator:add(nn.Sequential()
                      :add(self.rindexLinear)
                      :add(nn.LogSoftMax()))
    else
      generator:add(self:_simpleGeneratorLayer(rnn_size, sizes[i]))
    end
  end

  self:set(generator)
end

--[[ If the target vocabulary for the batch is not full vocabulary ]]
function Generator:setTargetVoc(t)
  self.rindexLinear:RIndex_setOutputIndices(t)
end

--[[ Release Generator for inference only ]]
function Generator:release()
  self.rindexLinear:RIndex_clean()
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
