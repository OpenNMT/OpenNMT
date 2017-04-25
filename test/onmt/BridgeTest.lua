require('onmt.init')

local tester = ...

local bridgeTest = torch.TestSuite()

function bridgeTest.inconsistentNumStates()
  tester:assertError(function () onmt.Bridge('copy', 500, 2, 500, 3) end)
end

function bridgeTest.inconsistentHiddenSize()
  tester:assertError(function () onmt.Bridge('copy', 500, 2, 400, 2) end)
end

function bridgeTest.copy()
  local bridge = onmt.Bridge('copy', 30, 2, 30, 2)

  local states = {
    torch.Tensor(2, 30):uniform(),
    torch.Tensor(2, 30):uniform()
  }
  local exp = onmt.utils.Tensor.recursiveClone(states)

  local outputs = bridge:forward(states)

  tester:eq(outputs, exp)
end

function bridgeTest.none()
  local bridge = onmt.Bridge('none', 30, 2, 30, 3)

  local states = {
    torch.Tensor(2, 30):uniform(),
    torch.Tensor(2, 30):uniform()
  }

  local outputs = bridge:forward(states)
  tester:eq(outputs, nil)
  local gradInputs = bridge:backward(states, outputs)
  tester:eq(gradInputs, nil)
end

function bridgeTest.dense()
  local bridge = onmt.Bridge('dense', 30, 2, 30, 3)

  local states = {
    torch.Tensor(2, 30):uniform(),
    torch.Tensor(2, 30):uniform()
  }

  local outputs = bridge:forward(states)

  tester:eq(#outputs, 3)
  tester:eq(outputs[1]:size(), torch.LongStorage({2, 30}))
end

function bridgeTest.dense_nonlinear()
  local bridge = onmt.Bridge('dense_nonlinear', 30, 2, 30, 3)

  local states = {
    torch.Tensor(2, 30):uniform(),
    torch.Tensor(2, 30):uniform()
  }

  local outputs = bridge:forward(states)

  tester:eq(#outputs, 3)
  tester:eq(outputs[1]:size(), torch.LongStorage({2, 30}))
end

function bridgeTest.loadBackwardCompatibility()
  -- Default bridge is copy.
  local bridge = onmt.Bridge.load()

  local states = {
    torch.Tensor(2, 30):uniform(),
    torch.Tensor(2, 30):uniform()
  }
  local exp = onmt.utils.Tensor.recursiveClone(states)

  local outputs = bridge:forward(states)

  tester:eq(outputs, exp)
end

return bridgeTest
