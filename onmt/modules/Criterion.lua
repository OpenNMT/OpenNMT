--[[ Default decoder generator. Given RNN state, produce categorical distribution.

Simply implements $$softmax(W h + b)$$.
--]]
local Criterion, parent = torch.class('onmt.Criterion', 'nn.ParallelCriterion')


function Criterion:__init(vocabSize, features, adaptive_softmax_cutoff)
  parent.__init(self, self:_buildCriterion(vocabSize, features, adaptive_softmax_cutoff))
end

function Criterion:_buildCriterion(vocabSize, features, adaptive_softmax_cutoff)
  local criterion = nn.ParallelCriterion(false)

  local function addNllCriterion(size)
    -- Ignores padding value.
    local w = torch.ones(size)
    w[onmt.Constants.PAD] = 0

    local nll = nn.ClassNLLCriterion(w)

    -- Let the training code manage loss normalization.
    nll.sizeAverage = false
    criterion:add(nll)
  end

  if adaptive_softmax_cutoff then
    criterion:add(nn.AdaptiveLoss( adaptive_softmax_cutoff ))
  else
    addNllCriterion(vocabSize)
  end

  for j = 1, #features do
    addNllCriterion(features[j]:size())
  end

  return criterion
end
