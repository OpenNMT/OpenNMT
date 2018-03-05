require('onmt.init')

local tester = ...

local sampledDatasetTest = torch.TestSuite()

function sampledDatasetTest.sample()

  _G.logger = onmt.utils.Logger.new(nil, true, nil)

  local dataSize = 1234
  local batchSize = 16

  local opt = {}
  opt.sample = 100
  opt.sample_perplexity = false
  opt.sample_perplexity_init = 100
  opt.sample_perplexity_max = 1000
  opt.sample_vocab = false
  opt.max_tokens = 100000

  local tds = require('tds')
  local srcData = {words = tds.Vec(), features = tds.Vec()}
  local tgtData = {words = tds.Vec(), features = tds.Vec()}
  for i = 1, dataSize do
    srcData.words:insert(torch.Tensor(5))
    srcData.features:insert(tds.Vec())
    srcData.features[i]:insert(torch.Tensor(5))
    local tgtSent = torch.LongTensor(5)
    for j = 1, 5 do
      tgtSent[j] = torch.random(1,1000)
    end
    tgtData.words:insert(tgtSent)
    tgtData.features:insert(tds.Vec())
    tgtData.features[i]:insert(torch.Tensor(5))
  end

  -- random sampling
  opt.sample_type = 'uniform'
  local dataset = onmt.data.SampledDataset.new(opt, srcData, tgtData)
  dataset:setBatchSize(batchSize)

  tester:eq(dataset:getNumSampled(), opt.sample)
  for i = 1, dataset:batchCount() do
    dataset:getBatch(i)
  end

  tester:eq(dataset.vocabTensor, nil)

  -- sampling with target vocabulary importance sampling
  opt.sample_vocab = true
  dataset = onmt.data.SampledDataset.new(opt, srcData, tgtData)
  dataset:setBatchSize(batchSize)
  dataset:getBatch(1)
  tester:assertgt(dataset.vocabTensor:size(1), 100)
  tester:assertle(dataset.vocabTensor:size(1), 500)

  -- sampling with ppl
  opt.sample_vocab = false
  opt.sample_type = 'perplexity'

  dataset = onmt.data.SampledDataset.new(opt, srcData, tgtData)
  dataset:setBatchSize(batchSize)

  tester:eq(dataset:getNumSampled(), opt.sample)
  for i = 1, dataset:batchCount() do
    dataset:getBatch(i)
  end

  -- oversampling
  opt.sample = 2000
  opt.sample_perplexity = false

  dataset = onmt.data.SampledDataset.new(opt, srcData, tgtData)
  dataset:setBatchSize(batchSize)

  tester:eq(dataset:getNumSampled(), opt.sample)
  for i = 1, dataset:batchCount() do
    dataset:getBatch(i)
  end

  -- partition sampling
  opt.sample_type = 'partition'
  opt.sample = 600
  dataset = onmt.data.SampledDataset.new(opt, srcData, tgtData)
  dataset:setBatchSize(batchSize)

  tester:eq(dataset:getNumSampled(), opt.sample)
  -- simulate 2 epochs
  for i = 1, dataset:batchCount() do
    dataset:getBatch(i)
  end
  for i = 1, dataset:batchCount() do
    dataset:getBatch(i)
  end

end

return sampledDatasetTest
