require('onmt.init')

local tester = ...

local positionEmbeddingTest = torch.TestSuite()

function positionEmbeddingTest.default()
  -- timesteps at dim 2, max position 3, embedding size	5
  local pos = onmt.PositionEmbedding(2, 3, 5)

  pos:postParametersInitialization()

  local input = torch.Tensor(2, 6, 5):uniform()
  local inputSize = torch.LongTensor({6, 3})
  local gradOutput = torch.Tensor(2, 6, 5):uniform()

  local output = pos:forward({input, inputSize})
  tester:eq(output:size(), torch.LongStorage({2, 6, 5}))

  -- test padding masking
  tester:eq(output[{2,{1,3}}], torch.Tensor(3,5):fill(0))

  -- test last positions
  tester:eq(output[{1,3}],output[{1,-1}])

  local gradInput = pos:backward({input, inputSize}, gradOutput)
  tester:eq(gradInput:size(), torch.LongStorage({2, 6, 5}))

end

return positionEmbeddingTest
