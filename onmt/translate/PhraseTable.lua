--[[Parse and lookup a words from a phrase table.
--]]
local PhraseTable = torch.class('PhraseTable')

function PhraseTable:__init(filePath)
  local f = assert(io.open(filePath, 'r'))

  self.table = {}

  for line in f:lines() do
    local c = onmt.utils.String.split(line, '|||')
    assert(#c == 2, 'badly formatted phrase table: ' .. line)
    self.table[onmt.utils.String.strip(c[1])] = onmt.utils.String.strip(c[2])
  end

  f:close()
end

--[[ Return the phrase table match for `word`. ]]
function PhraseTable:lookup(word)
  return self.table[word]
end

--[[ Return true if the phrase table contains the source word `word`. ]]
function PhraseTable:contains(word)
  return self:lookup(word) ~= nil
end

return PhraseTable
