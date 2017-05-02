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
    local feat_generator
    local linear
    if i == 1 and opt.target_voc_importance_sampling_size and opt.target_voc_importance_sampling_size > 0 then
      linear = onmt.RIndexLinear(rnn_size, sizes[i])
      self.rindexLinear = linear
    else
      linear = nn.Linear(rnn_size, sizes[i])
    end
    feat_generator = nn.Sequential()
                        :add(linear)
                        :add(nn.LogSoftMax())
    generator:add(feat_generator)
  end

  self:set(generator)
end

function Generator:setTargetVoc(tgtVec)
  if tgtVec and self.rindexLinear then
    self.rindexLinear:setOutputIndices(tgtVec)
  end
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
