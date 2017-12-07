--[[ Logger is a class used for maintaining logs in a log file.
--]]
local Logger = torch.class('Logger')
local levels =  { DEBUG = 0, INFO = 1, WARNING = 2, ERROR = 3, NONE = 4 }

local options = {
  {
    '-log_file', '',
    [[Output logs to a file under this path instead of stdout - if file name ending with json, output structure json.]]
  },
  {
    '-disable_logs', false,
    [[If set, output nothing.]]
  },
  {
    '-log_level', 'INFO',
    [[Output logs at this level and above.]],
    {
      enum = { 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'NONE' }
    }
  }
}

function Logger.declareOpts(cmd)
  cmd:setCmdLineOptions(options, 'Logger')
end

--[[ Construct a Logger object.

Parameters:
  * `logFile` - Outputs logs to a file under this path instead of stdout. ['']
  * `disableLogs` - If = true, output nothing. [false]
  * `logLevel` - Outputs logs at this level and above. ['INFO']

Example:

    logger = onmt.utils.Logger.new('log.txt')
    logger:info('%s is an extension of OpenNMT.', 'Im2Text')
    logger:shutDown()

]]
function Logger:__init(logFile, disableLogs, logLevel)
  logFile = logFile or ''
  self.jsonLog = logFile:sub(-5) == '.json'
  disableLogs = disableLogs or false
  logLevel = logLevel or 'INFO'

  self.mute = (logFile:len() > 0)
  if disableLogs then
    self:setVisibleLevel('NONE')
  else
    self:setVisibleLevel(logLevel)
  end
  if string.len(logFile) > 0 then
    self.logFile = io.open(logFile, (self.jsonLog and 'w') or 'a')
    if self.jsonLog then
      self.logFile:write("[\n");
    end
  else
    self.logFile = nil
  end
end

local function jsonize(msg)
  return msg:gsub("\\","\\\\"):gsub("\n","\\n"):gsub("\"","\\\"")
end

--[[ Log a message at a specified level.

Parameters:
  * `message` - the message to log.
  * `level` - the desired message level. ['INFO']

]]
function Logger:log(message, level)
  level = level or 'INFO'
  local timeStamp = os.date('%x %X')
  if self.jsonLog then
    if message:len() > 0 then
      self.logFile:write('["'..level..'","'..timeStamp..'","'..jsonize(message)..'"],\n')
      self.logFile:flush()
    end
  else
    local msgFormatted = string.format('[%s %s] %s', timeStamp, level, message)
    if (not self.mute) and self:_isVisible(level) then
      print (msgFormatted)
    end
    if self.logFile and self:_isVisible(level) then
      self.logFile:write(msgFormatted .. '\n')
      self.logFile:flush()
    end
  end
end

--[[ Log a message at 'INFO' level.

Parameters:
  * `message` - the message to log. Supports formatting string.

]]
function Logger:info(...)
  self:log(self:_format(...), 'INFO')
end

--[[ Log a message at 'WARNING' level.

Parameters:
  * `message` - the message to log. Supports formatting string.

]]
function Logger:warning(...)
  self:log(self:_format(...), 'WARNING')
end

--[[ Log a message at 'ERROR' level.

Parameters:
  * `message` - the message to log. Supports formatting string.

]]
function Logger:error(...)
  self:log(self:_format(...), 'ERROR')
end

--[[ Log a message at 'DEBUG' level.

Parameters:
  * `message` - the message to log. Supports formatting string.

]]
function Logger:debug(...)
  self:log(self:_format(...), 'DEBUG')
end


--[[ Set the visible message level. Lower level messages will be muted.

Parameters:
  * `level` - 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'NONE'.

]]
function Logger:setVisibleLevel(level)
  assert(levels[level])
  self.level = level
end

--[[ Private function for comparing level against visible level.

Parameters:
  * `level` - 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'NONE'.

]]
function Logger:_isVisible(level)
  return levels[level] >= levels[self.level]
end

function Logger:_format(...)
  if #table.pack(...) == 1 then
    return ...
  else
    return string.format(...)
  end
end

--[[ Deconstructor. Close the log file.
]]
function Logger:shutDown()
  if self.logFile then
    if self.jsonLog then
      self.logFile:write("[]]\n")
    end
    self.logFile:close()
  end
end

return Logger
