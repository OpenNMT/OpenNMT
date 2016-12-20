require 'class'
local path = require('pl.path')

local function logJsonRecursive(obj)
  if type(obj) == 'string' then
    io.write('"' .. obj .. '"')
  elseif type(obj) == 'table' then
    local first = true

    io.write('{')

    for key, val in pairs(obj) do
      if not first then
        io.write(',')
      else
        first = false
      end
      io.write('"' .. key .. '":')
      logJsonRecursive(val)
    end

    io.write('}')
  else
    io.write(tostring(obj))
  end
end

--[[ Recursively outputs a Lua object to a JSON objects followed by a new line. ]]
local function logJson(obj)
  logJsonRecursive(obj)
  io.write('\n')
end


--[[ logging is a class used for maintaining logs in a log file.
--]]
local logging = torch.class('onmt.logging')

--[[ Construct a logging object.
Parameters:
  * `logPath` - the path to log file.
  * `mute` - whether or not outputs to screen.
]]
function logging:__init(logPath, mute)
  mute = mute or false
  self.mute = mute
  local openMode = 'w'
  if path.exists(logPath) then
    local input = nil
    while not input do
      print('Logging file exits. Overwrite(o)? Append(a)? Abort(q)?')
      input = io.read()
      if input == 'o' or input == 'O' then
        openMode = 'w'
      elseif input == 'a' or input == 'A' then
        openMode = 'a'
      elseif input == 'q' or input == 'Q' then
        os.exit()
      else
        openMode = 'a'
      end
    end
  end
  self.logFile = io.open(logPath, openMode)
end

--[[ Log a message at a specified level.
Parameters:
  * `message` - the message to log.
  * `level` - the desired message level. ['INFO']
]]
function logging:log(message, level)
  level = level or 'INFO'
  local timeStamp = os.date('%x %X')
  local msgFormatted = string.format('[%s %s] %s', timeStamp, level, message)
  if not self.mute then
    print (msgFormatted)
  end
  if self.logFile then
    self.logFile:write(msgFormatted .. '\n')
    self.logFile:flush()
  end
end

--[[ Log a message at 'INFO' level.
Parameters:
  * `message` - the message to log. Supports formatting string.
]]
function logging:info(...)
  self:log(string.format(...), 'INFO')
end

--[[ Log a message at 'WARNING' level.
Parameters:
  * `message` - the message to log. Supports formatting string.
]]
function logging:warning(...)
  self:log(string.format(...), 'WARNING')
end

--[[ Log a message at 'ERROR' level.
Parameters:
  * `message` - the message to log. Supports formatting string.
]]
function logging:error(...)
  self:log(string.format(...), 'ERROR')
end

--[[ Deconstructor. Close the log file.
]]
function logging:shutDown()
  if self.logFile then
    self.logFile:close()
  end
end

return {
  logJson = logJson
}
