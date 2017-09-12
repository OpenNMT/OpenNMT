--[[ Dynamic Dataset class ]]

local DynamicDataset, _ = torch.class("DynamicDataset")

function DynamicDataset:__init(ddr)
  self.ddr = ddr
end

--[[ define batch size ]]
function DynamicDataset:setBatchSize(maxBatchSize, uneven_batches)
  -- time to build first sample
  local data = self.ddr.preprocessor:makeData('train', self.ddr.dicts)
  self.first = true
  if self.ddr.preprocessor.args.sample > 0 then
    self.dataset = onmt.data.SampledDataset.new(self.ddr.preprocessor.args, data.src, data.tgt)
  else
    self.dataset = onmt.data.Dataset.new(data.src, data.tgt)
  end
  self.maxBatchSize = maxBatchSize
  self.uneven_batches = uneven_batches
  local nTrainBatch, batchUsage = self.dataset:setBatchSize(maxBatchSize, uneven_batches)
  self.src = self.dataset.src
  self.maxSourceLength = self.dataset.maxSourceLength
  self.maxTargetLength = self.dataset.maxTargetLength
  return nTrainBatch, batchUsage
end

--[[ get a new sample ]]
function DynamicDataset:sample()
  if torch.type(self.dataset) == 'SampledDataset' then
    self.dataset:sample()
  elseif not self.first and self.ddr.preprocessor.args.gsample > 0 then
    _G.logger:info('Sampling dataset...')
    local data = self.ddr.preprocessor:makeData('train', self.ddr.dicts)
    self.dataset = onmt.data.Dataset.new(data.src, data.tgt)
    local nTrainBatch, _ = self.dataset:setBatchSize(self.maxBatchSize, self.uneven_batches)
    self.src = self.dataset.src
    self.maxSourceLength = self.dataset.maxSourceLength
    self.maxTargetLength = self.dataset.maxTargetLength
    _G.logger:info('Sampling completed - %d sentences, %d mini-batch', #self.src, nTrainBatch)
  else
    self.first = false
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
