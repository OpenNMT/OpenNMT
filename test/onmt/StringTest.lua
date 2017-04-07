require('onmt.init')

local tester = ...

local stringTest = torch.TestSuite()

function stringTest.split_noSplit()
  tester:eq(onmt.utils.String.split('foo-foo', '￨'), { 'foo-foo' })
end
function stringTest.split_emptySplit2()
  tester:eq(onmt.utils.String.split('￨', '￨'), { '', '' })
end
function stringTest.split_emptySplit1Right()
  tester:eq(onmt.utils.String.split('foo￨', '￨'), { 'foo', '' })
end
function stringTest.split_emptySplit1Middle()
  tester:eq(onmt.utils.String.split('foo￨￨bar', '￨'), { 'foo', '', 'bar' })
end
function stringTest.split_emptySplit1Left()
  tester:eq(onmt.utils.String.split('￨foo', '￨'), { '', 'foo' })
end
function stringTest.split_split2()
  tester:eq(onmt.utils.String.split('foo￨bar', '￨'), { 'foo', 'bar' })
end
function stringTest.split_split3()
  tester:eq(onmt.utils.String.split('foo￨bar￨foobar', '￨'), { 'foo', 'bar', 'foobar' })
end
function stringTest.split_ignoreEscaping1()
  tester:eq(onmt.utils.String.split('foo\\￨bar', '￨'), { 'foo\\', 'bar' })
end
function stringTest.split_ignoreEscaping2()
  tester:eq(onmt.utils.String.split('foo\\￨bar￨foobar', '￨'), { 'foo\\', 'bar', 'foobar' })
end
function stringTest.split_ignoreEscaping3()
  tester:eq(onmt.utils.String.split('\\￨', '￨'), { '\\', '' })
end
function stringTest.split_ignoreEscaping4()
  tester:eq(onmt.utils.String.split('\\\\￨N', '￨'), { '\\\\', 'N' })
end

function stringTest.strip_noStrip()
  tester:eq(onmt.utils.String.strip('foo'), 'foo')
end
function stringTest.strip_stripLeft()
  tester:eq(onmt.utils.String.strip('  foo'), 'foo')
end
function stringTest.strip_stripRight()
  tester:eq(onmt.utils.String.strip('foo  '), 'foo')
end
function stringTest.strip_stripBoth()
  tester:eq(onmt.utils.String.strip('    foo  '), 'foo')
end

return stringTest
