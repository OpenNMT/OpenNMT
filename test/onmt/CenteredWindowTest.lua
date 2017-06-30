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
  local cw = w[1]
  local muw = w[2]
  tester:eq(cw[1]:narrow(1,1,3):sum(), 0)
  tester:eq(cw[1][4]:sum(), 8)
  tester:eq(cw[2]:narrow(1,1,2):sum(), 0)
  tester:eq(cw[2][3]:sum(), 8)
  tester:eq(cw[12][7]:sum(), 0)
  tester:eq(cw[12][5]:sum(), 0)
  tester:eq(cw[12][4]:sum(), 8)
  tester:assertTensorEq(muw[1], muw[2])
  tester:eq(muw[1][4], 1)
end

function centeredWindowTest.forwardbackward()
  local c = torch.Tensor(12, 12, 8):fill(1)
  local p = torch.Tensor(12)
  for i = 1, 12 do
    p[i] = i
  end
  local gc = torch.Tensor(12, 7, 8):fill(1)
  local gmu = torch.Tensor(12, 7):uniform(0.1)
  local gI = onmt.CenteredWindow(3):backward({c, p}, {gc, gmu})

  local gI1 = gI[1]
  local gI2 = gI[2]
  tester:eq(gI1[1]:narrow(1,1,4):sum(), 32)
  tester:eq(gI1[1]:narrow(1,5,7):sum(), 0)
  tester:eq(gI1[5]:sum(), 7*8)
  tester:eq(gI2:sum(), 0)
end

return centeredWindowTest
