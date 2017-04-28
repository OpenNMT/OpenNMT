require('onmt.init')

local tester = ...

local phraseTableTest = torch.TestSuite()

local function dumpTable(tab, filename)
  local f = assert(io.open(filename, 'w'))
  for _, v in ipairs(tab) do
    f:write(v[1] .. '|||' .. v[2] .. '\n')
  end
  f:close()
end

function phraseTableTest.withSpaces()
  local tab = {
    { ' toto  ', '  tata' },
    { 'foo  ', '  bar  ' }
  }

  dumpTable(tab, 'pt.txt')

  local pt = onmt.translate.PhraseTable.new('pt.txt')

  tester:eq(pt:contains('foo'), true)
  tester:eq(pt:lookup('toto'), 'tata')
  tester:eq(pt:contains('toto'), true)
  tester:eq(pt:lookup('foo'), 'bar')

  os.remove('pt.txt')
end

return phraseTableTest
