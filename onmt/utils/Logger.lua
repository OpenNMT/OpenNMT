--[[ Logger is a class used for maintaining logs in a log file.
--]]
local Logger = torch.class('Logger')

function Logger.declareOpts(cmd)
  cmd:option('-log_file', '', [[Outputs logs to a file under this path instead of stdout.]])
  cmd:option('-disable_logs', false, [[If = true, output nothing.]])
end

--[[ Construct a Logger object.

Parameters:
  * `args` - options.

Example:

    local cmd = torch.CmdLine()
    onmt.utils.Logger.declareOpts(cmd)
    opt = cmd:parse(arg)
    logger = onmt.utils.Logger.new(opt)
    logger:info('%s is an extension of OpenNMT.', 'Im2Text')
    logger:shutDown()

]]
function Logger:__init(args)
  if args then
    local mute = (args.log_file:len() > 0)
    self:build(args.log_file, mute)
    if args.disable_logs then
      self:setVisibleLevel('ERROR')
    end
  end
end

--[[ Build a Logger object.

Parameters:
  * `logPath` - the path to log file.
  * `mute` - whether or not suppress outputs to stdout. [false]

Example:

    logger = onmt.utils.Logger.new():build("./log.txt")
    logger:info('%s is an extension of OpenNMT.', 'Im2Text')
    logger:shutDown()

]]
function Logger:build(logPath, mute)
  logPath = logPath or ''
  mute = mute or false
  self.mute = mute
  local openMode = 'w'
  local f = io.open(logPath, 'r')
  if f then
    f:close()
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
  if string.len(logPath) > 0 then
    self.logFile = io.open(logPath, openMode)
  else
    self.logFile = nil
  end
  self.LEVELS = { DEBUG = 0, INFO = 1, WARNING = 2, ERROR = 3 }
  return self
end

--[[ Log a message at a specified level.

Parameters:
  * `message` - the message to log.
  * `level` - the desired message level. ['INFO']

]]
function Logger:log(message, level)
  level = level or 'INFO'
  local timeStamp = os.date('%x %X')
  local msgFormatted = string.format('[%s %s] %s', timeStamp, level, message)
  if (not self.mute) and self:_isVisible(level) then
    print (msgFormatted)
  end
  if self.logFile and self:_isVisible(level) then
    self.logFile:write(msgFormatted .. '\n')
    self.logFile:flush()
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

--[[ Log a message as exactly it is.

Parameters:
  * `message` - the message to log. Supports formatting string.

]]
function Logger:write(...)
  local msg = self:_format(...)
  if (not self.mute) and self:_isVisible('WARNING') then
    io.write(msg)
  end
  if self.logFile and self:_isVisible('WARNING') then
    self.logFile:write(msg)
    self.logFile:flush()
  end
end

--[[ Set the visible message level. Lower level messages will be muted.

Parameters:
  * `level` - 'DEBUG', 'INFO', 'WARNING' or 'ERROR'.

]]
function Logger:setVisibleLevel(level)
  assert (level == 'DEBUG' or level == 'INFO' or
          level == 'WARNING' or level == 'ERROR')
  self.level = level
end

-- Private function for comparing level against visible level.
-- `level` - 'DEBUG', 'INFO', 'WARNING' or 'ERROR'.
function Logger:_isVisible(level)
  self.level = self.level or 'INFO'
  return self.LEVELS[level] >= self.LEVELS[self.level]
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
    self.logFile:close()
  end
end

return Logger
