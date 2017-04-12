return function(modelType)
  if modelType == 'seq2seq' then
    return onmt.Seq2Seq
  elseif modelType == 'lm' then
    return onmt.LanguageModel
  elseif modelType == 'seqtagger' then
    return onmt.SeqTagger
  else
    error('invalid model type ' .. modelType)
  end
end
