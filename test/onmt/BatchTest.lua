require('onmt.init')

local tester = ...

local batchTest = torch.TestSuite()

function batchTest.inconsistentSourceAndTarget()
  local src = {
    torch.IntTensor({5, 6, 7, 8}),
    torch.IntTensor({5, 6}),
    torch.IntTensor({5, 6, 7}),
    torch.IntTensor({5, 6, 7, 8}),
  }
  local tgt = {
    torch.IntTensor({onmt.Constants.BOS, 5, 6, 7, onmt.Constants.EOS}),
    torch.IntTensor({onmt.Constants.BOS, 5, 6, 7, 8, onmt.Constants.EOS}),
    torch.IntTensor({onmt.Constants.BOS, 5, 6, 7, 8, 9, onmt.Constants.EOS}),
  }

  tester:assertError(function () onmt.data.Batch.new(src, {}, tgt, {}) end)
end

function batchTest.simpleTrainingBatch()
  local src = {
    torch.IntTensor({5, 6, 7, 8}),
    torch.IntTensor({5, 6, 7, 8}),
    torch.IntTensor({5, 6, 7, 8}),
  }
  local tgt = {
    torch.IntTensor({onmt.Constants.BOS, 5, 6, 7, onmt.Constants.EOS}),
    torch.IntTensor({onmt.Constants.BOS, 5, 6, 7, 8, onmt.Constants.EOS}),
    torch.IntTensor({onmt.Constants.BOS, 5, 6, 7, 8, 9, onmt.Constants.EOS}),
  }

  local batch = onmt.data.Batch.new(src, {}, tgt, {})

  tester:eq(batch.size, 3)

  tester:eq(batch.sourceLength, 4)
  tester:eq(batch.sourceSize, torch.IntTensor({4, 4, 4}))
  tester:eq(batch.uneven, false)
  tester:eq(batch.sourceInput,
            torch.LongTensor({
              {5, 5, 5},
              {6, 6, 6},
              {7, 7, 7},
              {8, 8, 8}
            })
  )
  tester:eq(batch.sourceInputRev,
            torch.LongTensor({
              {8, 8, 8},
              {7, 7, 7},
              {6, 6, 6},
              {5, 5, 5}
            })
  )

  tester:eq(batch.targetLength, 5 + 1)
  tester:eq(batch.targetSize, torch.IntTensor({3 + 1, 4 + 1, 5 + 1}))
  tester:eq(batch.targetInput,
            torch.LongTensor({
              {onmt.Constants.BOS, onmt.Constants.BOS, onmt.Constants.BOS},
              {                 5,                  5,                  5},
              {                 6,                  6,                  6},
              {                 7,                  7,                  7},
              {onmt.Constants.PAD,                  8,                  8},
              {onmt.Constants.PAD, onmt.Constants.PAD,                  9}
            })
  )
  tester:eq(batch.targetOutput,
            torch.LongTensor({
              {                 5,                  5,                  5},
              {                 6,                  6,                  6},
              {                 7,                  7,                  7},
              {onmt.Constants.EOS,                  8,                  8},
              {onmt.Constants.PAD, onmt.Constants.EOS,                  9},
              {onmt.Constants.PAD, onmt.Constants.PAD, onmt.Constants.EOS}
            })
  )
end

function batchTest.simpleTrainingBatchUneven()
  local src = {
    torch.IntTensor({5, 6, 7, 8}),
    torch.IntTensor({5, 6}),
    torch.IntTensor({5, 6, 7}),
  }
  local tgt = {
    torch.IntTensor({onmt.Constants.BOS, 5, 6, 7, onmt.Constants.EOS}),
    torch.IntTensor({onmt.Constants.BOS, 5, 6, 7, 8, onmt.Constants.EOS}),
    torch.IntTensor({onmt.Constants.BOS, 5, 6, 7, 8, 9, onmt.Constants.EOS}),
  }

  local batch = onmt.data.Batch.new(src, {}, tgt, {})

  tester:eq(batch.sourceLength, 4)
  tester:eq(batch.sourceSize, torch.IntTensor({4, 2, 3}))
  tester:eq(batch.uneven, true)
  tester:eq(batch.sourceInput,
            torch.LongTensor({
              {5, onmt.Constants.PAD, onmt.Constants.PAD},
              {6, onmt.Constants.PAD,                  5},
              {7,                  5,                  6},
              {8,                  6,                  7}
            })
  )
  tester:eq(batch.sourceInputRev,
            torch.LongTensor({
              {8,                  6,                  7},
              {7,                  5,                  6},
              {6, onmt.Constants.PAD,                  5},
              {5, onmt.Constants.PAD, onmt.Constants.PAD}
            })
  )
end

function batchTest.trainingBatchWithFeatures()
  local src = {
    torch.IntTensor({5, 6, 7, 8}),
    torch.IntTensor({5, 6, 7}),
  }
  local srcFeatures = {
    {
      torch.IntTensor({10, 11, 12, 13}),
      torch.IntTensor({10, 11, 12, 13})
    },
    {
      torch.IntTensor({10, 11, 12}),
      torch.IntTensor({10, 11, 12})
    }
  }
  local tgt = {
    torch.IntTensor({onmt.Constants.BOS, 5, 6, 7, 8, 9, onmt.Constants.EOS}),
    torch.IntTensor({onmt.Constants.BOS, 5, 6, 7, 8, onmt.Constants.EOS}),
  }
  local tgtFeatures = {
    {
      torch.IntTensor({onmt.Constants.EOS, onmt.Constants.BOS, 10, 11, 12, 13, 14}),
      torch.IntTensor({onmt.Constants.EOS, onmt.Constants.BOS, 10, 11, 12, 13, 14})
    },
    {
      torch.IntTensor({onmt.Constants.EOS, onmt.Constants.BOS, 10, 11, 12, 13}),
      torch.IntTensor({onmt.Constants.EOS, onmt.Constants.BOS, 10, 11, 12, 13})
    }
  }

  local batch = onmt.data.Batch.new(src, srcFeatures, tgt, tgtFeatures)

  tester:eq(#batch.sourceInputFeatures, 2)
  tester:eq(#batch.sourceInputRevFeatures, 2)
  tester:eq(#batch.targetInputFeatures, 2)
  tester:eq(#batch.targetOutputFeatures, 2)

  local sourceInput3 = batch:getSourceInput(3)

  tester:eq(type(sourceInput3), 'table')
  tester:eq(#sourceInput3, 2)
  tester:eq(type(sourceInput3[2]), 'table')
  tester:eq(#sourceInput3[2], 2)

  local targetInput3 = batch:getTargetInput(3)

  tester:eq(type(targetInput3), 'table')
  tester:eq(#targetInput3, 2)
  tester:eq(type(targetInput3[2]), 'table')
  tester:eq(#targetInput3[2], 2)

  local targetOutput3 = batch:getTargetOutput(3)

  tester:eq(type(targetOutput3), 'table')
  tester:eq(#targetOutput3, 3)
end

function batchTest.batchQuery()
  local src = {
    torch.IntTensor({5, 6, 7, 8}),
    torch.IntTensor({5, 6, 7, 8}),
    torch.IntTensor({5, 6, 7, 8}),
  }
  local tgt = {
    torch.IntTensor({onmt.Constants.BOS, 5, 6, 7, onmt.Constants.EOS}),
    torch.IntTensor({onmt.Constants.BOS, 5, 6, 7, 8, onmt.Constants.EOS}),
    torch.IntTensor({onmt.Constants.BOS, 5, 6, 7, 8, 9, onmt.Constants.EOS}),
  }

  local batch = onmt.data.Batch.new(src, {}, tgt, {})

  tester:eq(batch:getSourceInput(2), torch.LongTensor({6, 6, 6}))
  tester:eq(batch:getTargetInput(2), torch.LongTensor({5, 5, 5}))
  tester:eq(batch:getTargetOutput(2), { torch.LongTensor({6, 6, 6}) })
end

function batchTest.simpleInferenceBatchUneven()
  local src = {
    torch.IntTensor({5, 6, 7, 8}),
    torch.IntTensor({5, 6}),
    torch.IntTensor({5, 6, 7}),
  }

  tester:assertNoError(function () onmt.data.Batch.new(src) end)
end

return batchTest
