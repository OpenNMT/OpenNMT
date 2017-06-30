require('onmt.init')

local tester = ...

local centeredWindowTest = torch.TestSuite()

function centeredWindowTest.forward()
  local c = torch.Tensor(12, 12, 8):fill(1)
  local p = torch.Tensor(12)
  for i = 1, 12 do
    p[i] = i
  end
  local w = onmt.CenteredWindow(3):forward({c, p})
  tester:eq(w[1]:narrow(1,1,3):sum(), 0)
  tester:eq(w[1][4]:sum(), 8)
  tester:eq(w[2]:narrow(1,1,2):sum(), 0)
  tester:eq(w[2][3]:sum(), 8)
  tester:eq(w[12][7]:sum(), 0)
  tester:eq(w[12][5]:sum(), 0)
  tester:eq(w[12][4]:sum(), 8)
end

return centeredWindowTest
