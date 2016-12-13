require 'torch'

--[[ Class for managing tab separated training logs ]]
local Log = torch.class("Log")

function Log:__init(logfile, dolog)
  self.logfile = logfile
  self.dolog = dolog
  if not self.dolog then return end
  local file = io.open(self.logfile, "a")
  if not file then
    print("ERROR: cannot access logfile", logfile)
    os.exit(0)
  end
  file:close()
end

function Log:clear()
  if not self.dolog then return end
  local file = io.open(self.logfile, "w")
  file:close()
end

function Log:append(L)
  if not self.dolog then return end
  local file = io.open(self.logfile, "a")
  for i = 1, #L do
    if i > 1 then file:write("\t") end
    file:write(L[i])
  end
  file:write("\n")
  file:close()
end

return Log
