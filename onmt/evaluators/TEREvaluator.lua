local TEREvaluator, parent = torch.class('TEREvaluator', 'TranslationEvaluator')

function TEREvaluator:__init(translatorOpt, dicts)
  parent.__init(self, translatorOpt, dicts)
end

function TEREvaluator:score(predictions, references)
  local bleu = onmt.scorers['ter'](predictions, { references })
  return bleu * 100
end

function TEREvaluator:compare(a, b, delta)
  return onmt.evaluators.Evaluator.lowerIsBetter(a, b, delta)
end

function TEREvaluator:__tostring__()
  return 'TER'
end

return TEREvaluator
