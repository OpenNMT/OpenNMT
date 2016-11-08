require 'torch'

local file_reader = torch.class("file_reader")

function file_reader:__init(filename)
  self.file = assert(io.open(filename, "r"))
end

function file_reader:next()
  local line = self.file:read()

  if line == nil then
    return nil
  end

  local sent = {}
  for word in line:gmatch'([^%s]+)' do
    table.insert(sent, word)
  end

  return sent
end

function file_reader:close()
  self.file:close()
end

return file_reader
