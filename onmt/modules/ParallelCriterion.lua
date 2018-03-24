--[[
  Define parallel ClassNLLCriterion.
--]]
local ParallelCriterion, parent = torch.class('onmt.ParallelCriterion', 'nn.ParallelCriterion')

function ParallelCriterion:__init(outputSizes)
  parent.__init(self, false)

  for i = 1, #outputSizes do
    -- only enable label_smoothing for tokens
    local nll = self:_addCriterion(outputSizes[i], i==1 and label_smoothing)
    if i == 1 then self.mainCriterion = nll end
  end
end

function Seq2Seq:setGeneratorVocabSize(size)
  self.mainCriterion.weights:resize(size)
end

function ParallelCriterion:_addCriterion(size, label_smoothing)
  label_smoothing = label_smoothing or 0
  local criterion

  if label_smoothing then
    -- Ignores padding value.
    local w = torch.ones(size)
    w[onmt.Constants.PAD] = 0
    criterion = nn.ClassNLLCriterion(w)
  else
    criterion = nn.DistKLDivCriterion()
    criterion.one_hot = torch.Tensor(1, size)
    criterion.one_hot:fill(label_smoothing / (len(size) - 2))
  end

  -- Let the training code manage loss normalization.
  criterion.sizeAverage = false
  self:add(criterion)
  return criterion
end
