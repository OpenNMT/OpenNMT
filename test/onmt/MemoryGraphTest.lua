require('onmt.init')

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

return memoryGraphTest
