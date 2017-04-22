require('onmt.init')

local tester = ...

local memoryOptimizerTest = torch.TestSuite()

local function testSequential(sequential, input, shareOutputs)
  local sequencer = onmt.Sequencer(sequential)
  sequencer:training()

  local memoryOptimizer = onmt.utils.MemoryOptimizer.new({ sequencer })

  local output = sequencer:net(1):forward(input)
  local _ = sequencer:net(1):backward(input, output:clone():uniform(-0.1, 0.1))

  memoryOptimizer:optimize()

  for i, m in ipairs(sequential.modules) do
    tester:eq(m.gradInputSharedIdx ~= nil, i > 1)
    tester:eq(m.outputSharedIdx ~= nil, shareOutputs[i])
  end
end

local function testModule(mod, input, shareInput, shareOutput)
  local sequential = nn.Sequential()
    :add(nn.Copy(nil, nil, true, true))
    :add(mod)
    :add(nn.Copy(nil, nil, true, true))

  testSequential(sequential, input, { shareInput, shareOutput, false })
end

function memoryOptimizerTest.exposedTensor()
  local sequential = nn.Sequential()
    :add(nn.Identity())

  testSequential(sequential, torch.Tensor(5), { false })
end

function memoryOptimizerTest.transferFunctions()
  local input = torch.Tensor(5):uniform(-0.1, 0.1)
  local transfers = {
    nn.Sigmoid(),
    nn.SoftMax(),
    nn.Tanh()
  }

  for _, t in ipairs(transfers) do
    testModule(t, input, true, false)
  end
end

function memoryOptimizerTest.linear()
  local input = torch.Tensor(5):uniform(-0.1, 0.1)
  testModule(nn.Linear(5, 10), input, false, true)
end

return memoryOptimizerTest
