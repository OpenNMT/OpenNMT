require('onmt.init')

local tester = torch.Tester()


local stringTest = torch.TestSuite()

function stringTest.noSplit()
  tester:eq(onmt.utils.String.split('foo-foo', '%-|%-'), { 'foo-foo' })
end
function stringTest.emptySplit2()
  tester:eq(onmt.utils.String.split('-|-', '%-|%-'), { '', '' })
end
function stringTest.emptySplit1Right()
  tester:eq(onmt.utils.String.split('foo-|-', '%-|%-'), { 'foo', '' })
end
function stringTest.emptySplit1Middle()
  tester:eq(onmt.utils.String.split('foo-|--|-bar', '%-|%-'), { 'foo', '', 'bar' })
end
function stringTest.emptySplit1Left()
  tester:eq(onmt.utils.String.split('-|-foo', '%-|%-'), { '', 'foo' })
end
function stringTest.split2()
  tester:eq(onmt.utils.String.split('foo-|-bar', '%-|%-'), { 'foo', 'bar' })
end
function stringTest.split3()
  tester:eq(onmt.utils.String.split('foo-|-bar-|-foobar', '%-|%-'), { 'foo', 'bar', 'foobar' })
end

function stringTest.noStrip()
  tester:eq(onmt.utils.String.strip('foo'), 'foo')
end
function stringTest.stripLeft()
  tester:eq(onmt.utils.String.strip('  foo'), 'foo')
end
function stringTest.stripRight()
  tester:eq(onmt.utils.String.strip('foo  '), 'foo')
end
function stringTest.stripBoth()
  tester:eq(onmt.utils.String.strip('    foo  '), 'foo')
end

tester:add(stringTest)


local nmttest = torch.TestSuite()

-- local function equal(t1, t2, msg)
--    if (torch.type(t1) == "table") then
--       for k, _ in pairs(t2) do
--          equal(t1[k], t2[k], msg)
--       end
--    else
--       tester:eq(t1, t2, 0.00001, msg)
--    end
-- end


function nmttest.Data()
end

tester:add(nmttest)

function onmt.test(tests, fixed_seed)
  -- Limit number of threads since everything is small
  local nThreads = torch.getnumthreads()
  torch.setnumthreads(1)

   -- Randomize stuff
  local seed = fixed_seed or (1e5 * torch.tic())
  print('Seed: ', seed)
  math.randomseed(seed)
  torch.manualSeed(seed)
  tester:run(tests)
  torch.setnumthreads(nThreads)
  return tester
end
