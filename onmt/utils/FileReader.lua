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
    while p<=#line and line:sub(p,p) == ' ' do
      p = p + 1
    end
    line = line:sub(p+1)
    if line == '' then
      line = self.file:read()
    end
    while true do
      local row = {}
      local hasEOS = false
      for tok in line:gmatch'([^%s]+)' do
        if tok == ']' then
          hasEOS=true
          break
        end
        table.insert(row, tok)
      end
      assert(hasEOS or #row ~= 0, 'Empty line in feature description: '..line)
      if #row > 0 then
        table.insert(sent, row)
      end
      if hasEOS then break end
      line = self.file:read()
    end
  end
  return sent, idx
end

function FileReader.countLines(filename)
  local BUFSIZE = 2^13
  local f = io.input(filename)
  local lc = 0
  while true do
    local lines, rest = f:read(BUFSIZE, "*line")
    if not lines then break end
    if rest then lines = lines .. rest .. '\n' end
    -- count newlines in the chunk
    local t
    t = select(2, string.gsub(lines, "\n", "\n"))
    lc = lc + t
  end
  return lc
end

function FileReader:close()
  self.file:close()
end

return FileReader
