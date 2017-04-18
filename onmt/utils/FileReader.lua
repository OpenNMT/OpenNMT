local FileReader = torch.class("FileReader")

function FileReader:__init(filename, idxSent, featSequence)
  self.file = assert(io.open(filename, "r"))
  self.idxSent = idxSent
  self.featSequence = featSequence
end

--[[ Read next line in the file and split it on spaces. If EOF is reached, returns nil. ]]
function FileReader:next()
  local line = self.file:read()
  local idx

  if line == nil then
    return nil
  end

  if self.idxSent then
    local p = line:find(" ")
    assert(p and p ~= 1, 'Invalid line - missing idx: '..line)
    idx = line:sub(1,p-1)
    line = line:sub(p+1)
  end

  local sent = {}
    if not self.featSequence then
    for word in line:gmatch'([^%s]+)' do
      table.insert(sent, word)
    end
  else
    local p = 1
    while p<=#line and line:sub(p,p) == ' ' do
      p = p + 1
    end
    assert(p <= #line and line:sub(p,p) == '[', 'Invalid feature start line (pos '..p..'): '..line)
    while true do
      line = self.file:read()
      local row = {}
      for tok in line:gmatch'([^%s]+)' do
        table.insert(row, tok)
      end
      assert(#row ~= 0, 'Empty line in feature description: '..line)
      if row[#row] == ']' then
        table.remove(row)
        if #row > 0 then
          table.insert(sent, row)
        end
        break
      end
      table.insert(sent, row)
    end
  end
  return sent, idx
end

function FileReader:close()
  self.file:close()
end

return FileReader
