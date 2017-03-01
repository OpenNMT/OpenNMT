require('onmt.init')

local tester = torch.Tester()


local stringTest = torch.TestSuite()

function stringTest.noSplit()
  tester:eq(onmt.utils.String.split('foo-foo', '￨'), { 'foo-foo' })
end
function stringTest.emptySplit2()
  tester:eq(onmt.utils.String.split('￨', '￨'), { '', '' })
end
function stringTest.emptySplit1Right()
  tester:eq(onmt.utils.String.split('foo￨', '￨'), { 'foo', '' })
end
function stringTest.emptySplit1Middle()
  tester:eq(onmt.utils.String.split('foo￨￨bar', '￨'), { 'foo', '', 'bar' })
end
function stringTest.emptySplit1Left()
  tester:eq(onmt.utils.String.split('￨foo', '￨'), { '', 'foo' })
end
function stringTest.split2()
  tester:eq(onmt.utils.String.split('foo￨bar', '￨'), { 'foo', 'bar' })
end
function stringTest.split3()
  tester:eq(onmt.utils.String.split('foo￨bar￨foobar', '￨'), { 'foo', 'bar', 'foobar' })
end
function stringTest.ignoreEscaping1()
  tester:eq(onmt.utils.String.split('foo\\￨bar', '￨'), { 'foo\\', 'bar' })
end
function stringTest.ignoreEscaping2()
  tester:eq(onmt.utils.String.split('foo\\￨bar￨foobar', '￨'), { 'foo\\', 'bar', 'foobar' })
end
function stringTest.ignoreEscaping3()
  tester:eq(onmt.utils.String.split('\\￨', '￨'), { '\\', '' })
end
function stringTest.ignoreEscaping4()
  tester:eq(onmt.utils.String.split('\\\\￨N', '￨'), { '\\\\', 'N' })
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

local profileTest = torch.TestSuite()

function profileTest.profiling()
  local profiler = onmt.utils.Profiler.new({profiler=true})
  profiler:start("main")
  local count = 0
  while count < 100 do count = count+1 end
  profiler:start("a")
  while count < 1000 do count = count+1 end
  profiler:stop("a"):start("b.c")
  while count < 10000 do count = count+1 end
  profiler:stop("b.c"):start("b.d"):stop("b.d")
  profiler:stop("main")
  local v=profiler:log():gsub("[-0-9.e]+","*")
  tester:assert(v=="main:{total:*,a:*,b:{total:*,d:*,c:*}}" or v == "main:{total:*,a:*,b:{total:*,c:*,d:*}}"
                or v == "main:{total:*,b:{total:*,c:*,d:*},a:*}" or v == "main:{total:*,b:{total:*,d:*,c:*},a:*}")
end

tester:add(profileTest)

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

tester:add(tensorTest)


local dictTest = torch.TestSuite()

function dictTest.createEmptyDict()
  local d = onmt.utils.Dict.new()
  tester:eq(d:size(), 0)
end

function dictTest.createSimpleDict()
  local d = onmt.utils.Dict.new()
  d:add('foo')
  d:add('bar')
  d:add('foobar')

  tester:eq(d:size(), 3)

  tester:ne(d:lookup('foo'), nil)
  tester:ne(d:lookup('bar'), nil)
  tester:ne(d:lookup('foobar'), nil)
end

function dictTest.createSimpleDictWithSpecialTokens()
  local d = onmt.utils.Dict.new({'toto', 'titi'})
  d:add('foo')
  d:add('bar')
  d:add('foobar')

  tester:eq(d:size(), 5)
  tester:eq(#d.special, 2)
  tester:ne(d:lookup('toto'), nil)
  tester:ne(d:lookup('titi'), nil)
end

function dictTest.createSimpleDictDuplicate()
  local d = onmt.utils.Dict.new()
  d:add('foo')
  d:add('bar')
  d:add('foobar')
  d:add('bar')

  tester:eq(d:size(), 3)
  tester:eq(d.frequencies[d:lookup('bar')], 2)
end

function dictTest.pruneLarger()
  local d = onmt.utils.Dict.new()
  d:add('foo')
  d:add('bar')
  d:add('foobar')
  d:add('bar')

  local pruned = d:prune(5)
  tester:eq(torch.pointer(pruned), torch.pointer(d))
end

function dictTest.pruneSmaller()
  local d = onmt.utils.Dict.new()
  d:add('foo')
  d:add('bar')
  d:add('foobar')
  d:add('bar')
  d:add('foobar')
  d:add('bar')

  local pruned = d:prune(2)
  tester:eq(pruned:size(), 2)
  tester:eq(pruned:lookup('foo'), nil)
  tester:ne(pruned:lookup('bar'), nil)
  tester:ne(pruned:lookup('foobar'), nil)
end

function dictTest.pruneSmallerWithSpecialTokens()
  local d = onmt.utils.Dict.new({ 'toto', 'titi' })
  d:add('foo')
  d:add('bar')
  d:add('foobar')
  d:add('bar')
  d:add('foobar')
  d:add('bar')

  local pruned = d:prune(2)
  tester:eq(pruned:size(), 2 + 2)
  tester:eq(pruned:lookup('foo'), nil)
  tester:ne(pruned:lookup('bar'), nil)
  tester:ne(pruned:lookup('foobar'), nil)
  tester:ne(pruned:lookup('toto'), nil)
  tester:ne(pruned:lookup('titi'), nil)
end

function dictTest.pruneInvariableSpecialTokensIndex()
  local d = onmt.utils.Dict.new({ 'toto', 'titi' })
  d:add('foo')
  d:add('bar')
  d:add('foobar')
  d:add('bar')
  d:add('foobar')
  d:add('bar')

  local totoIndex = d:lookup('toto')
  local titiIndex = d:lookup('titi')

  local pruned = d:prune(2)
  tester:eq(pruned:lookup('toto'), totoIndex)
  tester:eq(pruned:lookup('titi'), titiIndex)
end

tester:add(dictTest)

local beamSearchTest = torch.TestSuite()
function beamSearchTest.beamSearch()
  local transitionScores = { {-math.huge, math.log(.6), math.log(.4), -math.huge},
                   {math.log(.6), -math.huge, math.log(.4), -math.huge},
                   {-math.huge, -math.huge, math.log(.1), math.log(.9)},
                   {-math.huge, -math.huge, -math.huge, -math.huge}
               }
  transitionScores = torch.Tensor(transitionScores)

  local Advancer = onmt.translate.Advancer

  local initBeam = function()
    return onmt.translate.Beam.new(torch.LongTensor({1, 2, 3}), {})
  end
  local update = function()
  end
  local expand = function(beam)
    local tokens = beam:getTokens()
    local token = tokens[#tokens]
    local scores = transitionScores:index(1, token)
    return scores
  end
  local isComplete = function(beam)
    local tokens = beam:getTokens()
    local completed = tokens[#tokens]:eq(4)
    if #tokens - 1 > 2 then
      completed:fill(1)
    end
    return completed
  end

  Advancer.initBeam = function() return initBeam() end
  Advancer.update = function(_, beam) update(beam) end
  Advancer.expand = function(_, beam) return expand(beam) end
  Advancer.isComplete = function(_, beam) return isComplete(beam) end
  local beamSize, nBest, advancer, beamSearcher, results
  advancer = Advancer.new()
  -- Test different beam sizes
  nBest = 1
  -- Beam size 2
  beamSize = 2
  beamSearcher = onmt.translate.BeamSearcher.new(advancer)
  results = beamSearcher:search(beamSize, nBest)
  tester:eq(results, { {{tokens = {3, 4}, states = {}, score = math.log(.4*.9)}},
                       {{tokens = {3, 4}, states = {}, score = math.log(.4*.9)}},
                       {{tokens = {4}, states = {}, score = math.log(.9)}} }, 1e-6)
  -- Beam size 1
  beamSize = 1
  beamSearcher = onmt.translate.BeamSearcher.new(advancer)
  results = beamSearcher:search(beamSize, nBest)
  tester:eq(results, { {{tokens = {2, 1, 2}, states = {}, score = math.log(.6*.6*.6)}},
                       {{tokens = {1, 2, 1}, states = {}, score = math.log(.6*.6*.6)}},
                       {{tokens = {4}, states = {}, score = math.log(.9)}} }, 1e-6)

  -- Test nBest = 2
  nBest = 2
  beamSize = 3
  beamSearcher = onmt.translate.BeamSearcher.new(advancer)
  results = beamSearcher:search(beamSize, nBest)
  tester:eq(results, { {{tokens = {3, 4}, states = {}, score = math.log(.4*.9)},
                        {tokens = {2, 3, 4}, states = {}, score = math.log(.6*.4*.9)}},
                       {{tokens = {3, 4}, states = {}, score = math.log(.4*.9)},
                        {tokens = {1, 3, 4}, states = {}, score = math.log(.6*.4*.9)}},
                       {{tokens = {4}, states = {}, score = math.log(.9)},
                       {tokens = {3, 4}, states = {}, score = math.log(.1*.9)}} }, 1e-6)

  -- Test filter
  local filter = function(beam)
    local tokens = beam:getTokens()
    local batchSize = tokens[1]:size(1)
    -- Disallow {3, 4}
    local prune = torch.ByteTensor(batchSize):zero()
    for b = 1, batchSize do
      if #tokens >= 3 then
        if tokens[2][b] == 3 and tokens[3][b] == 4 then
          prune[b] = 1
        end
      end
    end
    return prune
  end
  Advancer.filter = function(_, beam) return filter(beam) end
  advancer = Advancer.new()
  nBest = 1
  beamSize = 3
  beamSearcher = onmt.translate.BeamSearcher.new(advancer)
  results = beamSearcher:search(beamSize, nBest)
  tester:eq(results, { {{tokens = {2, 3, 4}, states = {}, score = math.log(.6*.4*.9)}},
                       {{tokens = {1, 3, 4}, states = {}, score = math.log(.6*.4*.9)}},
                       {{tokens = {4}, states = {}, score = math.log(.9)}} }, 1e-6)
end

tester:add(beamSearchTest)


local SampledDatasetTest = torch.TestSuite()
function SampledDatasetTest.Sample()

  _G.logger = onmt.utils.Logger.new()

  local dataSize = 1234
  local samplingSize = 100
  local batchSize = 16
  local sample_w_ppl = false
  local sample_w_ppl_begin = 100
  local sample_w_ppl_max = 1000

  local tds = require('tds')
  local srcData = {words = tds.Vec(), features = tds.Vec()}
  local tgtData = {words = tds.Vec(), features = tds.Vec()}
  for i = 1, dataSize do
    srcData.words:insert(torch.Tensor(5))
    srcData.features:insert(tds.Vec())
    srcData.features[1]:insert(torch.Tensor(5))
    tgtData.words:insert(torch.Tensor(5))
    tgtData.features:insert(tds.Vec())
    tgtData.features[1]:insert(torch.Tensor(5))
  end

  tester:eq(#srcData.words, dataSize)

  local dataset = onmt.data.SampledDataset.new(srcData, tgtData, samplingSize, sample_w_ppl, sample_w_ppl_begin, sample_w_ppl_max)
  dataset:setBatchSize(batchSize)

  tester:eq(dataset:getBatch(1).size, 16)

  local numSampled = dataset:getNumSampled()
  local numBatch = math.ceil(numSampled / batchSize)

  tester:eq(dataset:batchCount(), numBatch)
  for i = 1, dataset:batchCount() do
    dataset:getBatch(i)
  end

  sample_w_ppl = true
  dataset = onmt.data.SampledDataset.new(srcData, tgtData, samplingSize, sample_w_ppl, sample_w_ppl_begin, sample_w_ppl_max)
  dataset:setBatchSize(batchSize)

  numSampled = dataset:getNumSampled()
  numBatch = math.ceil(numSampled / batchSize)

  tester:eq(dataset:getBatch(1).size, 16)
  tester:eq(dataset:batchCount(), numBatch)

  for i = 1, dataset:batchCount() do
    dataset:getBatch(i)
  end
end
tester:add(SampledDatasetTest)

local function main()
  -- Limit number of threads since everything is small
  local nThreads = torch.getnumthreads()
  torch.setnumthreads(1)

  tester:run()

  torch.setnumthreads(nThreads)

  if tester.errors[1] then
    os.exit(1)
  end
end

main()
