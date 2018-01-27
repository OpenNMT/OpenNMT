require('onmt.init')

local tester = ...

local placeholdersTest = torch.TestSuite()

function placeholdersTest.extractKV()
  local pack
  pack = table.pack(onmt.utils.Placeholders.norm("｟a：b｠"))
  tester:eq(pack, {"｟a｠","b", n=2})
  pack = table.pack(onmt.utils.Placeholders.norm("｟a｠"))
  tester:eq(pack, {"｟a｠", "｟a｠", n=2})
end

function placeholdersTest.protectedChar()
  local pack
  pack = table.pack(onmt.utils.Placeholders.norm("｟a：b％0020c｠"))
  tester:eq(pack, {"｟a｠","b c", n=2})
end

function placeholdersTest.extractTable()
  local keys,values = onmt.utils.Placeholders.norm({"｟a：123｠","｟b｠"})
  tester:eq(keys[1], "｟a｠")
  tester:eq(keys[2], "｟b｠")
  tester:eq(values[keys[1]], "123")
  tester:eq(values[keys[2]], "｟b｠")
end

return placeholdersTest
