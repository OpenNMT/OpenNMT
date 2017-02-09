return function(mtype)
  local modelClass
  if mtype == 'seq2seq' then
    require('onmt.models.Seq2Seq')
    modelClass = onmt.Model.Seq2Seq
  else
    require('onmt.models.LanguageModel')
    modelClass = onmt.Model.LanguageModel
  end
  return modelClass
end
