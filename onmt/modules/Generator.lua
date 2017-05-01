--[[ Default decoder generator.
     Given RNN state, produce categorical distribution for tokens and features

     Simply implements $$softmax(W h b)$$.
--]]
local Generator, parent = torch.class('onmt.Generator', 'onmt.Network')

-- for backward compatibility - still declare FeaturesGenerator - but no need to define it
torch.class('onmt.FeaturesGenerator', 'onmt.Network')

function Generator:__init(opt, sizes)
  parent.__init(self)
  self:_buildGenerator(opt, sizes)
end

function Generator:_buildGenerator(opt, sizes)
  local generator = nn.ConcatTable()
  local rnn_size = opt.rnn_size

  for i = 1, #sizes do
    local feat_generator
    local linear
    if i == 1 and opt.target_voc_importance_sampling_size > 0 then
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

function Generator:updateOutput(input)
  self.output = self.net:updateOutput(input)
  return self.output
end

function Generator:updateGradInput(input, gradOutput)
  self.gradInput = self.net:updateGradInput(input, gradOutput)
  return self.gradInput
end

function Generator:accGradParameters(input, gradOutput, scale)
  self.net:accGradParameters(input, gradOutput, scale)
end
