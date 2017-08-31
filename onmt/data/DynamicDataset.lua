--[[ Dynamic Dataset class ]]

local DynamicDataset, _ = torch.class("DynamicDataset")

function DynamicDataset:__init(ddr)
  self.ddr = ddr
  self.first = true
end

--[[ ]]
function DynamicDataset:setBatchSize(maxBatchSize, uneven_batches)
  -- time to build a sample
  local data = self.ddr.preprocessor:makeData('train', self.ddr.dicts)
  self.dataset = onmt.data.Dataset.new(data.src, data.tgt)
  self.src = self.dataset.src
  self.maxSourceLength = self.dataset.maxSourceLength
  self.maxTargetLength = self.dataset.maxTargetLength
  return self.dataset:setBatchSize(maxBatchSize, uneven_batches)
end

function DynamicDataset:sample()
  if not self.first and self.ddr.preprocessor.args.sample > 0 then
    _G.logger:log('Sampling dataset...')
    local data = self.ddr.preprocessor:makeData('train', self.ddr.dicts)
    self.dataset = onmt.data.Dataset.new(data.src, data.tgt)
    self.src = self.dataset.src
    self.maxSourceLength = self.dataset.maxSourceLength
    self.maxTargetLength = self.dataset.maxTargetLength
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

return DynamicDataset
