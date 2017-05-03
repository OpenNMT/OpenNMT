--[[
    Importance Sampling generator
--]]
local ISGenerator, parent = torch.class('onmt.ISGenerator', 'onmt.Generator')

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

function ISGenerator.declareOpts(cmd)
  cmd:setCmdLineOptions(options, 'Generator')
end

function ISGenerator:__init(opt, sizes)
  parent.__init(self, opt, sizes)
end

function ISGenerator:_buildGenerator(opt, sizes)
  local generator = nn.ConcatTable()
  local rnn_size = opt.rnn_size

  for i = 1, #sizes do
    if i == 1 then
      self.rindexLinear = onmt.RIndexLinear(rnn_size, sizes[i])
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
function ISGenerator:setTargetVoc(t)
  self.rindexLinear:setOutputIndices(t)
end

--[[ If the target vocabulary for the batch is not full vocabulary ]]
function ISGenerator:unsetTargetVoc(t)
  self.rindexLinear:unsetOutputIndices(t)
end

--[[ Release Generator for inference only ]]
function ISGenerator:release()
  self.net:replace(function(m)
    if torch.type(m) == 'onmt.RIndexLinear' then
      local l = nn.Linear(m.fullWeight:size(2), m.fullWeight:size(1), m.fullBias)
      l.weight = m.weight
      l.bias = m.bias
      return l
    else
      return m
    end
  end)
end
