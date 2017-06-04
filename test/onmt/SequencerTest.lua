require('onmt.init')

local tester = ...

local sequencerTest = torch.TestSuite()

function sequencerTest.unroll()
  local net = nn.Linear(10, 20)
  local sequencer = onmt.Sequencer(net)

  sequencer:training()

  local _ = sequencer:net(2)

  tester:eq(#sequencer.networkClones, 2)

  for _, n in ipairs(sequencer.networkClones) do
    tester:ne(torch.pointer(n), torch.pointer(net))
    tester:eq(torch.pointer(n.weight:storage()), torch.pointer(net.weight:storage()))
  end
end

function sequencerTest.unrollWithSharing()
  local net = nn.Linear(10, 20)
  net.gradInput = torch.Tensor(10)
  net.output = torch.Tensor(20)
  net.gradInputSharedIdx = 1
  net.outputSharedIdx = 2

  local sequencer = onmt.Sequencer(net)

  sequencer:training()

  local net1 = sequencer:net(1)
  local net2 = sequencer:net(2)

  tester:ne(torch.pointer(net2), torch.pointer(net1))
  tester:eq(torch.pointer(net2.gradInput:storage()), torch.pointer(net1.gradInput:storage()))
  tester:eq(torch.pointer(net2.output:storage()), torch.pointer(net1.output:storage()))
end

function sequencerTest.inference()
  local net = nn.Linear(10, 20)
  local sequencer = onmt.Sequencer(net)

  sequencer:evaluate()

  local net5 = sequencer:net(5)

  tester:eq(#sequencer.networkClones, 0)
  tester:eq(torch.pointer(net5), torch.pointer(net))
end


function sequencerTest.evaluate()
  local net = nn.Linear(10, 20)
  local sequencer = onmt.Sequencer(net)

  sequencer:training()

  local net1 = sequencer:net(1)
  local _ = sequencer:net(2)

  sequencer:evaluate()

  tester:eq(torch.pointer(sequencer:net(5)), torch.pointer(net1))
  tester:eq(net1.train, false)
end

return sequencerTest
