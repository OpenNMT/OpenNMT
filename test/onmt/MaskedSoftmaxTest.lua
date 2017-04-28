require('onmt.init')

local tester = ...

local maskedSoftmaxTest = torch.TestSuite()

function maskedSoftmaxTest.none()
  local standard = nn.SoftMax()
  local masked = onmt.MaskedSoftmax(torch.LongTensor({4, 4}), 4)

  local input = torch.Tensor(2, 4):uniform()

  tester:eq(standard:forward(input), masked:forward(input), 1e-8)
end

function maskedSoftmaxTest.mask()
  local standard = nn.SoftMax()

  local sizes = {3, 2, 4}
  local max = 4
  local masked = onmt.MaskedSoftmax(torch.LongTensor(sizes), max)

  local input = torch.Tensor(#sizes, max):uniform()

  local maskedOutput = masked:forward(input)

  for i = 1, input:size(1) do
    tester:eq(maskedOutput[i]:sum(), 1, 1e-3)
    tester:eq(maskedOutput[i]:narrow(1, max - sizes[i] + 1, sizes[i]),
              standard:forward(input[i]:narrow(1, max - sizes[i] + 1, sizes[i])),
              1e-8)
  end
end

return maskedSoftmaxTest
