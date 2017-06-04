require('onmt.init')
require 'nngraph'
local tester = ...

local memoryGraphTest = torch.TestSuite()

local function buildEncoder(class)
  local cmd = onmt.utils.ExtendedCmdLine.new()
  class.declareOpts(cmd)

  local opt = cmd:parse('')

  local inputNet = nn.LookupTable(10, 4)
  inputNet.inputSize = 4

  return class(opt, inputNet), opt
end

function memoryGraphTest.read()

  local encoder = buildEncoder(onmt.Encoder)

  local MG = onmt.utils.MemoryGraph.new()

  MG:add("encoder", encoder.network)
  MG:dump('.')

  local file = io.open('encoder.dot')

  tester:assert(file ~= nil)

  file:close()
  os.remove('encoder.dot')
end

function memoryGraphTest.checkProtected()
  local m = nn.Linear(50,50)
  tester:assert(onmt.utils.MemoryGraph.protected(m, "input")==true)
  tester:assert(onmt.utils.MemoryGraph.protected(m, "output")==false)
end

function memoryGraphTest.graph1()

  local i = nn.Identity()()
  local x = nn.Linear(10,10)(i)
  x = nn.Dropout(0.2)(x)
  local o = nn.Linear(10,10)(x)
  local g1 = nn.gModule({i}, {o})

  local MG = onmt.utils.MemoryGraph.new()
  MG:add("graph1", g1)
  MG:dump(".")

  local t1 = torch.DoubleTensor(10)
  local u1 = g1:forward(t1):clone()
  g1:backward(t1, u1)

  local totalSize, protectedSize, saveSize = MG:optimize()

  -- totalSize: 3 x gradInput, 1 input, 1 output, 2 input-output
  tester:eq(totalSize, 560)
  -- protectedSize: 2xLinearInput, 1 output
  tester:eq(protectedSize, 240)
  -- saveSize: 2xgradInput + dropout input cluster => 2 save
  tester:eq(saveSize, 160)
end

function memoryGraphTest.graph2x1()
  local i = nn.Identity()()
  local x = nn.Linear(10,10)(i)
  x = nn.Dropout(0.2)(x)
  local o = nn.Linear(10,10)(x)
  local g1 = nn.gModule({i}, {o})

  -- second identical graph
  i = nn.Identity()()
  x = nn.Linear(10,10)(i)
  x = nn.Dropout(0.2)(x)
  o = nn.Linear(10,10)(x)
  local g2 = nn.gModule({i}, {o})

  local MG = onmt.utils.MemoryGraph.new()
  MG:add("graph1", g1)
  MG:add("graph2", g2)
  MG:dump(".")

  local t1 = torch.DoubleTensor(10)
  local u1 = g1:forward(t1):clone()
  g1:backward(t1, u1)

  local t2 = torch.DoubleTensor(10)
  local u2 = g2:forward(t2):clone()
  g2:backward(t2, u2)

  local totalSize, protectedSize, saveSize = MG:optimize()

  -- totalSize: 3 x gradInput, 1 input, 1 output, 2 input-output
  tester:eq(totalSize, 1120)
  -- protectedSize: 2xLinearInput, 1 output
  tester:eq(protectedSize, 480)
  -- 2 clusters of 3 + 5 => 6 saved
  tester:eq(saveSize, 480)
end

return memoryGraphTest
