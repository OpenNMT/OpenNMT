local LossEvaluator, parent = torch.class('LossEvaluator', 'Evaluator')

function LossEvaluator:__init()
  parent.__init(self)
end

function LossEvaluator:eval(model, data)
  local loss = 0
  local totalWords = 0

  for i = 1, data:batchCount() do
    local batch = onmt.utils.Cuda.convert(data:getBatch(i))
    loss = loss + model:forwardComputeLoss(batch)
    totalWords = totalWords + model:getOutputLabelsCount(batch)
  end

  return loss / totalWords
end

function LossEvaluator:compare(a, b, delta)
  return onmt.evaluators.Evaluator.lowerIsBetter(a, b, delta)
end

function LossEvaluator:__tostring__()
  return 'loss'
end

return LossEvaluator
