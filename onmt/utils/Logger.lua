--[[ Logger is a class used for maintaining logs in a log file.
--]]
local Logger = torch.class('Logger')

--[[ Construct a Logger object.

Parameters:
  * `logPath` - the path to log file.
  * `mute` - whether or not suppress outputs to stdout. [false]

Example:

    logger = onmt.utils.Logger.new("./log.txt")
    logger:info('%s is an extension of OpenNMT.', 'Im2Text')
    logger:shutDown()

]]
function Logger:__init(logPath, mute)
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
function Logger:info(...)
  self:log(string.format(...), 'INFO')
end

--[[ Log a message at 'WARNING' level.

Parameters:
  * `message` - the message to log. Supports formatting string.

]]
function Logger:warning(...)
  self:log(string.format(...), 'WARNING')
end

--[[ Log a message at 'ERROR' level.

Parameters:
  * `message` - the message to log. Supports formatting string.

]]
function Logger:error(...)
  self:log(string.format(...), 'ERROR')
end

--[[ Log a message as exactly it is.

Parameters:
  * `message` - the message to log. Supports formatting string.

]]
function Logger:write(...)
  local msg = string.format(...)
  if not self.mute then
    print (msg)
  end
  if self.logFile then
    self.logFile:write(msg)
    self.logFile:flush()
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
