--[[ Dynamic Dataset class ]]

local DynamicDataset, _ = torch.class("DynamicDataset", "SampledVocabDataset")

function DynamicDataset:__init(opt, ddr)
  self.ddr = ddr
  self.opt = opt
end

--[[ define batch size ]]
function DynamicDataset:setBatchSize(maxBatchSize, maxTokens, uneven_batches)
  -- time to build first sample
  local data = self.ddr.preprocessor:makeData('train', self.ddr.dicts)
  self.first = true
  if self.ddr.preprocessor.args.sample > 0 then
    self.dataset = onmt.data.SampledDataset.new(self.ddr.preprocessor.args, data.src, data.tgt)
  else
    self.dataset = onmt.data.Dataset.new(data.src, data.tgt)
  end
  self.maxBatchSize = maxBatchSize
  self.maxTokens = maxTokens
  self.uneven_batches = uneven_batches
  local nTrainBatch, batchUsage = self.dataset:setBatchSize(maxBatchSize, maxTokens, uneven_batches)
  self.src = self.dataset.src
  self.tgt = self.dataset.tgt
  self.maxSourceLength = self.dataset.maxSourceLength
  self.maxTargetLength = self.dataset.maxTargetLength
  if torch.type(self.dataset) ~= 'SampledDataset' then
    self:sampleVocabInit(self.opt, self.src, self.tgt)
  end
  return nTrainBatch, batchUsage
end

--[[ get a new sample ]]
function DynamicDataset:sample()
  if torch.type(self.dataset) == 'SampledDataset' then
    self.dataset:sample()
  elseif self.ddr.preprocessor.args.gsample > 0 then
    if not self.first then
      _G.logger:info('Sampling dataset...')
      local data = self.ddr.preprocessor:makeData('train', self.ddr.dicts)
      self.dataset = onmt.data.Dataset.new(data.src, data.tgt)
      self.src = self.dataset.src
      self.tgt = self.dataset.tgt
      self:sampleVocabInit(self.opt, self.src, self.tgt)
      local nTrainBatch, _ = self.dataset:setBatchSize(self.maxBatchSize, self.maxTokens, self.uneven_batches)
      self.maxSourceLength = self.dataset.maxSourceLength
      self.maxTargetLength = self.dataset.maxTargetLength
      _G.logger:info('Sampling completed - %d sentences, %d mini-batch', #self.src, nTrainBatch)
    else
      self.first = false
    end
    if self.vocabIndex then
      -- if importance sampling select vocabs
      self:sampleVocabClear()
      for idx = 1, #self.vocabAxis do
        self:selectVocabs(idx)
      end
      self:sampleVocabReport('INFO')
      -- adapt vocab index to sampled Vocab
      for idx = 1, #self.vocabAxis do
        for j = 1, self.vocabAxis[idx]:size(1) do
          self.vocabAxis[idx][j] = self:sampleVocabIdx(self.vocabAxisName, self.vocabAxis[idx][j])
        end
      end
    end

  end
end

function DynamicDataset:batchCount()
  return self.dataset:batchCount()
end

function DynamicDataset:instanceCount()
  return self.dataset:instanceCount()
end

function DynamicDataset:getBatch(idx)
  return self.dataset:getBatch(idx)
end

function DynamicDataset:checkModel(model)
  if self.dataset.checkModel then
    self.dataset:checkModel(model)
  end
end

return DynamicDataset
