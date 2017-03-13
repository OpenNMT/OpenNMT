return function(modelType)
  if modelType == 'seq2seq' then
    return onmt.Seq2Seq
  elseif modelType == 'lm' then
    return onmt.LanguageModel
  else
    error('invalid model type ' .. modelType)
  end
end
