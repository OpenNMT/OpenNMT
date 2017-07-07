local DLratioEvaluator, parent = torch.class('DLratioEvaluator', 'TranslationEvaluator')

function DLratioEvaluator:__init(translatorOpt, dicts)
  parent.__init(self, translatorOpt, dicts)
end

function DLratioEvaluator:score(predictions, references)
  local DLratio = onmt.scorers['dlratio'](predictions, references)
  return DLratio * 100
end

function DLratioEvaluator:compare(a, b, delta)
  return onmt.evaluators.Evaluator.lowerIsBetter(a, b, delta)
end

function DLratioEvaluator:__tostring__()
  return 'DLratio'
end

return DLratioEvaluator
