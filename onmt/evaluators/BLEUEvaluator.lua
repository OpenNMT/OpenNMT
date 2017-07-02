local BLEUEvaluator, parent = torch.class('BLEUEvaluator', 'TranslationEvaluator')

function BLEUEvaluator:__init(translatorOpt, dicts)
  parent.__init(self, translatorOpt, dicts)
end

function BLEUEvaluator:score(predictions, references)
  local bleu = onmt.scorers['bleu'](predictions, { references })
  return bleu * 100
end

function BLEUEvaluator:compare(a, b, delta)
  return onmt.evaluators.Evaluator.higherIsBetter(a, b, delta)
end

function BLEUEvaluator:__tostring__()
  return 'BLEU'
end

return BLEUEvaluator
