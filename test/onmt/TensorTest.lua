require('onmt.init')

local tester = ...

local tensorTest = torch.TestSuite()

function tensorTest.reuseSmaller()
  local a = torch.Tensor(10, 200)
  local b = onmt.utils.Tensor.reuseTensor(a, { 5, 200 })
  tester:eq(torch.pointer(b:storage()), torch.pointer(a:storage()))
end

function tensorTest.reuseSame()
  local a = torch.Tensor(10, 200)
  local b = onmt.utils.Tensor.reuseTensor(a, a:size())
  tester:eq(torch.pointer(b:storage()), torch.pointer(a:storage()))
end

function tensorTest.reuseMultipleResize()
  local a = torch.Tensor(10, 200)
  local b = onmt.utils.Tensor.reuseTensor(a, { 5, 200 })
  tester:eq(torch.pointer(b:storage()), torch.pointer(a:storage()))
  local c = onmt.utils.Tensor.reuseTensor(a, { 10, 200 })
  tester:eq(torch.pointer(c:storage()), torch.pointer(a:storage()))
end

return tensorTest
