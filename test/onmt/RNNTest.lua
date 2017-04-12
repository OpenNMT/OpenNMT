require('onmt.init')

local tester = ...

local rnnTest = torch.TestSuite()

local function checkModuleCount(obj, name, exp)
  local count = 0
  obj:apply(function (m)
    if torch.typename(m) == name then
      count = count + 1
    end
  end)
  tester:eq(count, exp)
end

local function buildStates(count, batchSize, dim)
  local states = {}
  for _ = 1, count do
    table.insert(states, torch.Tensor(batchSize, dim):zero())
  end
  return states
end

local function testRNN(cell, layers, inputSize, hiddenSize, dropout, residual, dropout_input)
  local rnn = cell(layers, inputSize, hiddenSize, dropout, residual, dropout_input)
  local numStates = torch.typename(rnn) == 'onmt.GRU' and 1 or 2
  local inputs = buildStates(layers * numStates, 2, hiddenSize)
  table.insert(inputs, torch.Tensor(2, inputSize):uniform())

  local expectedDropout = 0
  if dropout and dropout > 0 then
    expectedDropout = expectedDropout + layers - 1
    if dropout_input then
      expectedDropout = expectedDropout + 1
    end
  end

  checkModuleCount(rnn, 'nn.Dropout', expectedDropout)

  local outputs = rnn:forward(inputs)

  if type(outputs) ~= 'table' then
    outputs = { outputs }
  end

  tester:eq(#outputs, layers * numStates)
  for i = 1, #outputs do
    tester:eq(outputs[i]:size(), torch.LongStorage({2, hiddenSize}))
  end
end


function rnnTest.LSTM_oneLayer()
  testRNN(onmt.LSTM, 1, 10, 20)
end
function rnnTest.LSTM_oneLayerWithInputDropout()
  testRNN(onmt.LSTM, 1, 10, 20, 0.3, false, true)
end
function rnnTest.LSTM_oneLayerWithoutInputDropout()
  testRNN(onmt.LSTM, 1, 10, 20, 0, false, true)
end
function rnnTest.LSTM_twoLayers()
  testRNN(onmt.LSTM, 2, 10, 20)
end
function rnnTest.LSTM_twoLayersWithDropout()
  testRNN(onmt.LSTM, 2, 10, 20, 0.3)
end
function rnnTest.LSTM_twoLayersWithInputDropout()
  testRNN(onmt.LSTM, 2, 10, 20, 0.3, false, true)
end

function rnnTest.GRU_oneLayer()
  testRNN(onmt.GRU, 1, 10, 20)
end
function rnnTest.GRU_oneLayerWithInputDropout()
  testRNN(onmt.GRU, 1, 10, 20, 0.3, false, true)
end
function rnnTest.GRU_oneLayerWithoutInputDropout()
  testRNN(onmt.LSTM, 1, 10, 20, 0, false, true)
end
function rnnTest.GRU_twoLayers()
  testRNN(onmt.GRU, 2, 10, 20)
end
function rnnTest.GRU_twoLayersWithDropout()
  testRNN(onmt.GRU, 2, 10, 20, 0.3)
end
function rnnTest.GRU_twoLayersWithInputDropout()
  testRNN(onmt.GRU, 2, 10, 20, 0.3, false, true)
end

return rnnTest
