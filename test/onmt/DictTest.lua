require('onmt.init')

local path = require('pl.path')

local tester = ...

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

function dictTest.pruneByMinFrequency()
  local d = onmt.utils.Dict.new({ 'toto', 'titi' })
  d:add('foo')
  d:add('bar')
  d:add('foobar')
  d:add('bar')
  d:add('foobar')
  d:add('bar')

  local totoIndex = d:lookup('toto')
  local titiIndex = d:lookup('titi')

  local pruned = d:pruneByMinFrequency(2)
  tester:eq(pruned:lookup('toto'), totoIndex)
  tester:eq(pruned:lookup('titi'), titiIndex)
  tester:ne(pruned:lookup('bar'), nil)
  tester:ne(pruned:lookup('foobar'), nil)
  tester:eq(pruned:lookup('foo'), nil)
end

function dictTest.saveAndLoad()
  local d1 = onmt.utils.Dict.new()
  d1:add('foo')
  d1:add('bar')
  d1:add('foobar')
  d1:add('toto')

  d1:writeFile('tmp.dict')
  tester:ne(path.exists('tmp.dict'), false)

  local d2 = onmt.utils.Dict.new('tmp.dict')

  tester:eq(d2:size(), d1:size())
  tester:eq(d2.idxToLabel, d1.idxToLabel)
  tester:eq(d2.labelToIdx, d1.labelToIdx)

  os.remove('tmp.dict')
end

return dictTest
