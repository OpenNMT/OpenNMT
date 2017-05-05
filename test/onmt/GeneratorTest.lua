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

function generatorTest.approximate()
  local opt = {}
  opt.rnn_size = 100
  local sizes = { 100, 5 }
  local generator = onmt.Generator.new(opt, sizes)

  local context = torch.Tensor(opt.rnn_size)
  local output = generator:forward({context, torch.Tensor{3}})

  tester:eq(#output, #sizes)
  for i = 1, #sizes do
    tester:eq( output[i]:dim(), 1)
    tester:eq( output[i]:size(1), sizes[i])
  end
end

function generatorTest.GeneratorIS()
  local opt = {}
  opt.rnn_size = 100
  local sizes = { 100, 5 }
  local generator = onmt.Generator.new(opt, sizes)
  generator:apply(function(m) m:training() end)

  local context = torch.Tensor(10, opt.rnn_size)
  local output = generator:forward({context, torch.Tensor{3}})

  tester:eq(#output, #sizes)
  for i = 1, #sizes do
    tester:eq( output[i]:dim(), 2)
    tester:eq( output[i]:size(2), sizes[i])
  end

  generator:setTargetVoc(torch.LongTensor{1,2})
  local output_ri = generator:forward({context, torch.Tensor{3}})

  print('output=',output)
  print('output_ri=',output_ri)
  tester:eq(output[1]:narrow(2,1,2), output_ri[1]:narrow(2,1,2))
  tester:eq(output[2], output_ri[2])
  tester:eq(output_ri[1]:size(2),2)
end

return generatorTest
