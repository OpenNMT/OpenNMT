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

    local referenceTargetsBatch = batch.targetOutput
    local referenceFeaturesBatch = batch.targetOutputFeatures
    local referenceSize = batch.targetSize

    batch:removeTarget()

    local predictedTargets, predictedFeatures, _, attention = translator:translateBatch(batch)

    for b = 1, batch.size do
      local referenceWords = self.dicts.tgt.words:convertToLabels(
        torch.totable(referenceTargetsBatch[{{}, b}]),
        onmt.Constants.EOS)
      table.remove(referenceWords) -- Remove </s>.

      local predictedWords = translator:buildTargetWords(predictedTargets[b][1],
                                                         referenceWords,
                                                         attention)

      table.insert(predictions, predictedWords)
      table.insert(references, referenceWords)

      if saveFile then
        local score = self:score({ predictedWords }, { referenceWords })

        -- When saving to a file, build the complete translation.
        local output = translator:buildOutput({
           words = predictedWords,
          features = translator:buildTargetFeatures(predictedFeatures[b][1])
        })

        -- Also build the reference sentence.
        local referenceFeatures = {}
        for l = 1, referenceSize[b] do
          local features = {}
          for j = 1, #referenceFeaturesBatch do
            table.insert(features, referenceFeaturesBatch[j][l][b])
          end
          table.insert(referenceFeatures, features)
        end

        local gold = translator:buildOutput({
          words = referenceWords,
          features = translator:buildTargetFeatures(referenceFeatures)
        })

        file:write(tostring(score))
        file:write(' ||| ')
        file:write(output)
        file:write(' ||| ')
        file:write(gold)
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
