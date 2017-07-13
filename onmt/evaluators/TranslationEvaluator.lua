local TranslationEvaluator, parent = torch.class('TranslationEvaluator', 'Evaluator')

function TranslationEvaluator:__init(translatorOpt, dicts)
  parent.__init(self)
  self.translatorOpt = translatorOpt
  self.dicts = dicts
end

function TranslationEvaluator:canSaveTranslation()
  return true
end

function TranslationEvaluator:eval(model, data, saveFile)
  local translator = onmt.translate.Translator.new(self.translatorOpt, model, self.dicts)

  local references = {}
  local predictions = {}

  local file

  if saveFile then
    file = assert(io.open(saveFile, 'w'))
  end

  for i = 1, data:batchCount() do
    local batch = onmt.utils.Cuda.convert(data:getBatch(i))

    local referenceTargets = batch.targetOutput
    batch:removeTarget()
    local predictedTargets, predictedFeatures, _, attention = translator:translateBatch(batch)

    for b = 1, batch.size do
      local reference = self.dicts.tgt.words:convertToLabels(torch.totable(referenceTargets[{{}, b}]),
                                                             onmt.Constants.EOS)
      table.remove(reference) -- Remove </s>.

      local targetWords = translator:buildTargetWords(predictedTargets[b][1], reference, attention)

      table.insert(predictions, targetWords)
      table.insert(references, reference)

      if saveFile then
        local score = self:score({ targetWords }, { reference })

        -- When saving to a file, build the complete translation.
        local targetFeatures = translator:buildTargetFeatures(predictedFeatures[b][1])

        local output = translator:buildOutput({
          words = targetWords,
          features = targetFeatures
        })

        file:write(tostring(score))
        file:write(' ||| ')
        file:write(output)
        file:write('\n')
      end
    end
  end

  if saveFile then
    file:close()
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
