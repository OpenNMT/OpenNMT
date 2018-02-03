require('onmt.init')

local tester = ...

local datasetTest = torch.TestSuite()

function datasetTest.trainingDataset()
  local srcData = {
    features = {},
    words = {
      torch.IntTensor(2),
      torch.IntTensor(2),
      torch.IntTensor(2),
      torch.IntTensor(3),
      torch.IntTensor(3),
      torch.IntTensor(4),
      torch.IntTensor(4),
      torch.IntTensor(4),
      torch.IntTensor(4),
      torch.IntTensor(4),
      torch.IntTensor(5),
    }
  }

  local tgtData = {
    features = {},
    words = {
      torch.IntTensor(4),
      torch.IntTensor(6),
      torch.IntTensor(5),
      torch.IntTensor(8),
      torch.IntTensor(5),
      torch.IntTensor(9),
      torch.IntTensor(9),
      torch.IntTensor(7),
      torch.IntTensor(6),
      torch.IntTensor(5),
      torch.IntTensor(8),
    }
  }

  local dataset = onmt.data.Dataset.new(srcData, tgtData)

  local batch = dataset:getBatch(1)
  tester:eq(batch.size, 11)

  local count, _ = dataset:setBatchSize(1, 500)
  tester:eq(count, 11)
  tester:eq(dataset.maxSourceLength, 5)
  tester:eq(dataset.maxTargetLength, 9 - 1)

  count, _ = dataset:setBatchSize(2, 500)
  tester:eq(count, 7)
  count, _ = dataset:setBatchSize(3, 500)
  tester:eq(count, 5)

  batch = dataset:getBatch()
  tester:eq(batch.size, 11)
  batch = dataset:getBatch(1)
  tester:eq(batch.size, 3)

  count, _ = dataset:setBatchSize(3, 500, true)
  tester:eq(count, 4)

  batch = dataset:getBatch(4)
  tester:eq(batch.size, 2)
end

function datasetTest.inferenceDataset()
  local srcData = {
    features = {},
    words = {
      torch.IntTensor(2),
      torch.IntTensor(2),
      torch.IntTensor(2),
      torch.IntTensor(3),
      torch.IntTensor(3),
      torch.IntTensor(4),
      torch.IntTensor(4),
      torch.IntTensor(4),
      torch.IntTensor(4),
      torch.IntTensor(4),
      torch.IntTensor(5),
    }
  }

  local dataset = onmt.data.Dataset.new(srcData)

  local count, _ = dataset:setBatchSize(2, 5)
  tester:eq(count, 7)
  tester:eq(dataset.maxSourceLength, 5)

  local batch = dataset:getBatch(1)
  tester:eq(batch.size, 2)
end

return datasetTest
