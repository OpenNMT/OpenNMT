local TranslationEvaluator, parent = torch.class('TranslationEvaluator', 'Evaluator')

function TranslationEvaluator:__init(translatorOpt, dicts)
  parent.__init(self)
  self.translatorOpt = translatorOpt
  self.dicts = dicts
end

function TranslationEvaluator:eval(model, data)
  local translator = onmt.translate.Translator.new(self.translatorOpt, model, self.dicts)

  local references = {}
  local predictions = {}

  for i = 1, data:batchCount() do
    local batch = onmt.utils.Cuda.convert(data:getBatch(i))

    local referenceTargets = batch.targetOutput
    batch:removeTarget()
    local predictedTargets = translator:translateBatch(batch)

    for b = 1, batch.size do
      local predicted = self.dicts.tgt.words:convertToLabels(predictedTargets[b][1])
      local reference = self.dicts.tgt.words:convertToLabels(torch.totable(referenceTargets[{{}, b}]),
                                                             onmt.Constants.EOS)
      table.remove(reference) -- Remove </s>.

      table.insert(predictions, predicted)
      table.insert(references, reference)
    end
  end

  return self:score(predictions, references)
end

--[[ Score translations against references.

Parameters:

  * `predictions` - a table of predicted sentences.
  * `references` - a table of reference sentences.

Returns: the score (e.g. BLEU).

]]
function TranslationEvaluator:score(_, _)
  error('Not implemented')
end

return TranslationEvaluator
