local PhraseTable = torch.class('PhraseTable')

function PhraseTable:__init(file_path)
  local f = assert(io.open(file_path, 'r'))

  self.table = {}

  for line in f:lines() do
    local c = line:split("|||")
    self.table[utils.String.strip(c[1])] = c[2]
  end

  f:close()
end

function PhraseTable:lookup(word)
  return self.table[word]
end

function PhraseTable:contains(word)
  return self:lookup(word) ~= nil
end

return PhraseTable
