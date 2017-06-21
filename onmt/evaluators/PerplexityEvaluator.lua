local PerplexityEvaluator, parent = torch.class('PerplexityEvaluator', 'LossEvaluator')

function PerplexityEvaluator:__init()
  parent.__init(self)
end

function PerplexityEvaluator:eval(model, data)
  return math.exp(parent.eval(self, model, data))
end

function PerplexityEvaluator:__tostring__()
  return 'perplexity'
end

return PerplexityEvaluator
