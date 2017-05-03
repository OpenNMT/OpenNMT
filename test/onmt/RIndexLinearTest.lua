require('onmt.init')

local tester = ...

local rindexLinearTest = torch.TestSuite()

function rindexLinearTest.regular()
  local m_standard = nn.Linear(5,20)
  m_standard:getParameters():uniform(-0.1, 0.1)
  local m_rindex = onmt.RIndexLinear(5,20)
  m_rindex.weight = m_standard.weight
  m_rindex.bias = m_standard.bias
  local input = torch.Tensor(5):uniform()
  tester:eq(m_standard:forward(input), m_rindex:forward(input), 1e-8)
end

function rindexLinearTest.inferenceTensor()
  local m_standard = nn.Linear(5,20)
  m_standard:getParameters():uniform(-0.1, 0.1)
  local m_rindex = onmt.RIndexLinear(5,20)
  m_rindex.weight = m_standard.weight
  m_rindex.bias = m_standard.bias

  m_rindex:setOutputIndices(torch.LongTensor{3,5,12})
  m_rindex:setOutputIndices()

  local input = torch.Tensor(5):uniform()
  tester:eq(m_standard:forward(input), m_rindex:forward(input), 1e-8)
end

function rindexLinearTest.inferenceTensorBatch()
  local m_standard = nn.Linear(5,20)
  m_standard:getParameters():uniform(-0.1, 0.1)
  local m_rindex = onmt.RIndexLinear(5,20)
  m_rindex.weight = m_standard.weight
  m_rindex.bias = m_standard.bias

  m_rindex:setOutputIndices(torch.LongTensor{3,5,12})
  m_rindex:setOutputIndices()

  local input = torch.Tensor(8, 5):uniform()
  tester:eq(m_standard:forward(input), m_rindex:forward(input), 1e-8)
end

function rindexLinearTest.trainingTensor()
  local m_standard = nn.Linear(5,20)
  m_standard:getParameters():uniform(-0.1, 0.1)
  local m_rindex = onmt.RIndexLinear(5,20)
  m_rindex.fullWeight:copy(m_standard.weight)
  m_rindex.fullBias:copy(m_standard.bias)

  m_rindex:setOutputIndices(torch.LongTensor{3,5,12})

  local input = torch.Tensor(5):uniform()
  local ri_output = m_rindex:forward(input)
  local std_output = m_standard:forward(input)

  tester:eq(ri_output[1], std_output[3], 1e-8)

  ri_output:uniform(0.1)
  local gradInput = m_rindex:backward(input, ri_output)
  tester:eq(gradInput:size(), input:size())

end

return rindexLinearTest
