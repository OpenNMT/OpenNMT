--[[
  Define parallel ClassNLLCriterion.
--]]
local ParallelCriterion, parent = torch.class('onmt.ParallelCriterion', 'nn.ParallelCriterion')

function ParallelCriterion:__init(outputSizes, label_smoothing)
  parent.__init(self, false)

  for i = 1, #outputSizes do
    -- only enable label_smoothing for tokens
    local nll = self:_addCriterion(outputSizes[i], i==1 and label_smoothing)
    if i == 1 then self.mainCriterion = nll end
  end
end

function ParallelCriterion:updateVocab(vocab)
  if self.mainCriterion.updateVocab then
    self.mainCriterion:updateVocab(vocab)
  else
    self.mainCriterion.weights:resize(vocab:size(1))
  end
end

function ParallelCriterion:_addCriterion(size, label_smoothing)
  label_smoothing = label_smoothing or 0
  local criterion

  if not label_smoothing then
    -- Ignores padding value.
    local w = torch.ones(size)
    w[onmt.Constants.PAD] = 0
    criterion = nn.ClassNLLCriterion(w)
  else
    criterion = onmt.LabelSmoothingCriterion(size, label_smoothing)
  end

  -- Let the training code manage loss normalization.
  criterion.sizeAverage = false
  self:add(criterion)
  return criterion
end