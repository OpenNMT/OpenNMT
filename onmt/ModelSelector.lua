return function(mtype)
  local modelClass
  if mtype == 'seq2seq' then
    modelClass = onmt.Seq2Seq
  else
    modelClass = onmt.LanguageModel
  end
  return modelClass
end
