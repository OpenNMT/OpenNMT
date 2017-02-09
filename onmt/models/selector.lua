return function(mtype)
  local modelClass
  if mtype == 'seq2seq' then
    require('onmt.models.seq2seq')
    modelClass = onmt.Model.seq2seq
  else
    require('onmt.models.LM')
    modelClass = onmt.Model.LM
  end
  return modelClass
end
