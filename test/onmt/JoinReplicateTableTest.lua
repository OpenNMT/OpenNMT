require('onmt.init')

local tester = ...

local joinReplicateTest = torch.TestSuite()

function joinReplicateTest.forwardbackward()
  local m = onmt.JoinReplicateTable(2, 3)
  local t1 = torch.Tensor(64, 1, 13)
  local t2 = torch.Tensor(64, 10, 13)
  local o = m:forward({t1, t2})
  tester:assert(o:dim() == 3)
  tester:assert(o:size(2) == 10)
  o:uniform(0.1)
  local gi = m:backward({t1, t2}, o)
  tester:assert(gi[1]:dim() == 3)
  tester:assert(gi[1]:size(2) == 1)
  tester:assert(gi[2]:dim() == 3)
  tester:assert(gi[2]:size(2) == 10)
end

return joinReplicateTest
