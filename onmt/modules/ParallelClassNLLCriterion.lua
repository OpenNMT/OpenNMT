--[[
  Define parallel ClassNLLCriterion.
--]]
local ParallelClassNLLCriterion, parent = torch.class('onmt.ParallelClassNLLCriterion', 'nn.ParallelCriterion')

function ParallelClassNLLCriterion:__init(outputSizes)
  parent.__init(self, false)

  for i = 1, #outputSizes do
    local nll = self:_addCriterion(outputSizes[i])
    if i == 1 then self.mainCriterion = nll end
  end
end

function ParallelClassNLLCriterion:_addCriterion(size)
  -- Ignores padding value.
  local w = torch.ones(size)
  w[onmt.Constants.PAD] = 0

  local nll = nn.ClassNLLCriterion(w)

  -- Let the training code manage loss normalization.
  nll.sizeAverage = false
  self:add(nll)
  return nll
end
