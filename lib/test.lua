require('lib.onmt.init')
require('lib.data')

local mytester = torch.Tester()

local nmttest = torch.TestSuite()

-- local function equal(t1, t2, msg)
--    if (torch.type(t1) == "table") then
--       for k, _ in pairs(t2) do
--          equal(t1[k], t2[k], msg)
--       end
--    else
--       mytester:eq(t1, t2, 0.00001, msg)
--    end
-- end


function nmttest.Data()
end

mytester:add(nmttest)

function onmt.test(tests, fixed_seed)
  -- Limit number of threads since everything is small
  local nThreads = torch.getnumthreads()
   torch.setnumthreads(1)

   -- Randomize stuff
   local seed = fixed_seed or (1e5 * torch.tic())
   print('Seed: ', seed)
   math.randomseed(seed)
   torch.manualSeed(seed)
   mytester:run(tests)
   torch.setnumthreads(nThreads)
   return mytester
end
