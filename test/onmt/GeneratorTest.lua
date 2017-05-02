require('onmt.init')

local tester = ...

local generatorTest = torch.TestSuite()

function generatorTest.backward_Compatibility()
  local opt = {}
  opt.rnn_size = 100
  local generator = onmt.Generator.new(opt,{10})

  generator.version = nil
  generator:set(nn.Sequential():add(nn.Linear(opt.rnn_size, 10)):add(nn.LogSoftMax()))

  local ngenerator = onmt.Generator.load(generator)
  tester:assertge(ngenerator.version, 2)
  tester:eq(type(generator:forward(torch.Tensor(opt.rnn_size))),'table')

end


return generatorTest
