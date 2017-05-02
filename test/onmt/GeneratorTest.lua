require('onmt.init')

local tester = ...

local generatorTest = torch.TestSuite()

function generatorTest.features()
  local opt = {}
  opt.rnn_size = 100
  local sizes = { 100, 5 }
  local generator = onmt.Generator.new(opt, sizes)

  local context = torch.Tensor(opt.rnn_size)
  local output = generator:forward(context)

  tester:eq(#output, #sizes)
  for i = 1, #sizes do
    tester:eq( output[i]:dim(), 1)
    tester:eq( output[i]:size(1), sizes[i])
  end
end

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
