--[[
  Training Criterion
--]]
local Criterion, parent = torch.class('onmt.Criterion', 'nn.ParallelCriterion')


function Criterion:__init(vocabSize, features)
  parent.__init(self, false)
  self:_buildCriterion(vocabSize, features)
end

function Criterion:_buildCriterion(vocabSize, features)
  local function addNllCriterion(size)
    -- Ignores padding value.
    local w = torch.ones(size)
    w[onmt.Constants.PAD] = 0

    local nll = nn.ClassNLLCriterion(w)

    -- Let the training code manage loss normalization.
    nll.sizeAverage = false
    self:add(nll)
  end

  addNllCriterion(vocabSize)

  for j = 1, #features do
    addNllCriterion(features[j]:size())
  end
end
