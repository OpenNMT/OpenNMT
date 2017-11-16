local normalizer = torch.class('normalizer')

local paths = require 'paths'

function normalizer:__init(cmd)
  self.cmd = cmd
end

function normalizer:normalize(lines)
  local input
  if type(lines) == "table" then
    input = table.concat(lines, "\n")
  else
    input = lines
  end
  local name = paths.tmpname ()
  local f = io.open(name, "w")
  f:write(input)
  f:close()
  local fout = io.popen("cat " .. name .. " | " .. self.cmd)
  local out = {}
  while true do
    local line = fout:read("*l")
    if not line then break end
    table.insert(out, line)
  end
  fout:close()
  os.remove(name)
  if type(lines) == "table" then
    if #out ~= #lines then
      return nil
    end
    return out
  else
    if #out ~= 1 then
      return nil
    end
    return out[1]
  end
end

return normalizer
