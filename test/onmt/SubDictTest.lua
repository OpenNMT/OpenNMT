require('onmt.init')

local tester = ...

local subDictTest = torch.TestSuite()

function subDictTest.read()
  local d1 = onmt.utils.Dict.new()
  d1:add(onmt.Constants.PAD_WORD)
  d1:add(onmt.Constants.UNK_WORD)
  d1:add(onmt.Constants.BOS_WORD)
  d1:add(onmt.Constants.EOS_WORD)
  d1:add('foo')
  d1:add('bar')
  d1:add('foobar')
  d1:add('toto')

  local f = io.open ("tmp.subdict", "w")
  f:write("bar\nfoobar\n")
  f:close()

  local sd1 = onmt.utils.SubDict.new(d1, 'tmp.subdict')

  tester:eq(sd1.targetVocTensor, torch.LongTensor{2,3,4,6,7})
  local t = torch.LongTensor{4}
  sd1:fullIdx(t)
  tester:assertTensorEq(t, torch.LongTensor{6})

  os.remove('tmp.subdict')
end

return subDictTest
