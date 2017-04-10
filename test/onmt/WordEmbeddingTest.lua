require('onmt.init')

local tester = ...

local wordEmbeddingTest = torch.TestSuite()

function wordEmbeddingTest.default()
  local emb = onmt.WordEmbedding(10, 5)
  local input = torch.LongTensor({{1, 2}, {3, 4}, {5, 6}})
  local output = emb:forward(input)

  tester:eq(output:size(), torch.LongStorage({3, 2, 5}))
end


local function testFixedEmb(emb)
  tester:eq(emb:parameters(), nil)
  tester:eq(emb.net.gradWeight, nil)

  local weights = emb.net.weight:clone()

  local input = torch.LongTensor({{1, 2}, {3, 4}, {5, 6}})
  local gradOutput = torch.Tensor(3, 2, 5):uniform()

  local _ = emb:forward(input)
  local _ = emb:backward(input, gradOutput)

  tester:eq(emb.net.weight, weights)
end

function wordEmbeddingTest.fixedAtConstruction()
  local emb = onmt.WordEmbedding(10, 5, '', true)
  testFixedEmb(emb)
end

function wordEmbeddingTest.fixedToogled()
  local emb = onmt.WordEmbedding(10, 5)
  emb:fixEmbeddings(true)
  testFixedEmb(emb)
end

return wordEmbeddingTest
