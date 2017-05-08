require('onmt.init')

local tester = ...

local featuresEmbeddingTest = torch.TestSuite()

function featuresEmbeddingTest.oneFeatConcat()
  local featEmb = onmt.FeaturesEmbedding({10}, {5}, 'concat')

  tester:eq(featEmb.outputSize, 5)

  local input = torch.LongTensor({1, 2})
  local output = featEmb:forward(input)

  tester:eq(output:size(), torch.LongStorage({2, 5}))
end

function featuresEmbeddingTest.oneFeatSum()
  local featEmb = onmt.FeaturesEmbedding({10}, {5}, 'sum')

  tester:eq(featEmb.outputSize, 5)

  local input = torch.LongTensor({1, 2})
  local output = featEmb:forward(input)

  tester:eq(output:size(), torch.LongStorage({2, 5}))
end

function featuresEmbeddingTest.twoFeatsSumDifferentSize()
  tester:assertError(function () onmt.FeaturesEmbedding({10, 20}, {5, 6}, 'sum') end)
end

function featuresEmbeddingTest.twoFeatsSum()
  local featEmb = onmt.FeaturesEmbedding({10, 20}, {5, 5}, 'sum')

  tester:eq(featEmb.outputSize, 5)

  local input = { torch.LongTensor({1, 2}), torch.LongTensor({3, 4}) }
  local output = featEmb:forward(input)

  tester:eq(torch.isTensor(output), true)
  tester:eq(output:size(), torch.LongStorage({2, 5}))
end

function featuresEmbeddingTest.twoFeatsConcat()
  local featEmb = onmt.FeaturesEmbedding({10, 20}, {5, 6}, 'concat')

  tester:eq(featEmb.outputSize, 5 + 6)

  local input = { torch.LongTensor({1, 2}), torch.LongTensor({3, 4}) }
  local output = featEmb:forward(input)

  tester:eq(torch.isTensor(output), true)
  tester:eq(output:size(), torch.LongStorage({2, 5 + 6}))
end

return featuresEmbeddingTest
