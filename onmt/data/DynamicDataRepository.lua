--[[ Global Data Repository class ]]

local DynamicDataRepository, _ = torch.class("DynamicDataRepository")

function DynamicDataRepository.declareOpts(cmd, modelClass)
  local data_type = modelClass.dataType()
  onmt.data.Preprocessor.declareOpts(cmd, data_type)
end

function DynamicDataRepository:__init(args, modelClass)
  self.dataType = modelClass.dataType()
  self.preprocessor = onmt.data.Preprocessor.new(args, self.dataType)
  self.dicts = self.preprocessor:getVocabulary()
  self.args = args
end

function DynamicDataRepository:getTraining()
  return onmt.data.DynamicDataset.new(self.args, self)
end

function DynamicDataRepository:getValid()
  local data = self.preprocessor:makeData('valid', self.dicts)
  if data then
    return onmt.data.Dataset.new(data.src, data.tgt)
  end
end

return DynamicDataRepository
