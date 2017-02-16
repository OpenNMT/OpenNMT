------------------------------------------------------------------------------------------------------------------
-- Local utility functions
------------------------------------------------------------------------------------------------------------------

--[[ Convert `val` string to its actual type (boolean, number or string). ]]
local function convert(key, val, ref)
  local new

  if type(val) == type(ref) then
    new = val
  elseif type(ref) == 'boolean' then
    if val == 'true' then
      new = true
    elseif val == 'false' then
      new = false
    else
      error('option ' .. key .. ' expects a boolean value [true, false]')
    end
  elseif type(ref) == 'number' then
    new = tonumber(val)
    assert(new ~= nil, 'option ' .. key .. ' expects a number value')
  end

  return new
end

------------------------------------------------------------------------------------------------------------------

local ExtendedCmdLine, parent, path
-- the utils function can be run without torch
if torch then
  ExtendedCmdLine, parent = torch.class('ExtendedCmdLine', 'torch.CmdLine')
  path = require('pl.path')
else
  ExtendedCmdLine = {}
end

--[[
  Extended handling of command line options - provide validation methods, and utilities for handling options
  at module level.

  For end-user - provides '-h' for help, with possible '-md' variant for preparing MD ready help file.

  Provides also possibility to read options from config file with '-config file' or save current options to
  config file with '-save_config file'

  Example:

    cmd = onmt.utils.ExtendedCmdLine.new()
    local optim_options = {
      {'-max_batch_size',     64   , 'Maximum batch size', {valid=onmt.utils.ExtendedCmdLine.isUInt()}}
    }

    cmd:setCmdLineOptions(optim_options, 'Optimization')
    local opt = cmd:parse(arg)

    local optimArgs = cmd.getModuleOpts(opt, optim_options)
]]

function ExtendedCmdLine:__init(script)
  self.script = script
  parent.__init(self)

  self:text("")
  self:option('-h', false, 'this help file')
  self:option('-config', '', 'read options from config file.', {valid=ExtendedCmdLine.fileNullOrExists})
  self:option('-save_config', '', 'save options from config file.')

end

function ExtendedCmdLine:help(arg, doMd)
  if doMd then
    for _,option in ipairs(self.helplines) do
      if type(option) == 'table' then
        io.write('* ')
        if option.default ~= nil then -- it is an option
          io.write("`"..option.key.."`: ")
          if option.meta and option.meta.enum then
            io.write(' ('..table.concat(option.meta.enum, ', ')..') ')
          end
          option.help = option.help:gsub(" *\n   *"," ")
          if option.help then io.write(option.help) end
          io.write(' [' .. tostring(option.default) .. ']')
        else -- it is an argument
          io.write('<' .. onmt.utils.String.stripHyphens(option.key) .. '>')
          if option.help then io.write(' ' .. option.help) end
        end
      else
        local display = option:gsub("%*","-")
        io.write(display) -- just some additional help
      end
      io.write('\n')
    end
    io.write('\n')
  else
    if arg then
      io.write('Usage: ')
      io.write(arg[0]..' ' .. self.script .. ' ')
      io.write('[options] ')
      for i=1,#self.arguments do
        io.write('<' .. onmt.utils.String.stripHyphens(self.arguments[i].key) .. '>')
      end
      io.write('\n')
    end

    -- first pass to compute max length
    local optsz = 0
    for _,option in ipairs(self.helplines) do
      if type(option) == 'table' then
        if option.default ~= nil then -- it is an option
          if #option.key > optsz then
            optsz = #option.key
          end
        else -- it is an argument
          local stripOptionKey = onmt.utils.String.stripHyphens(option.key)
          if #stripOptionKey+2 > optsz then
            optsz = #stripOptionKey+2
          end
        end
      end
    end

    local padMultiLine = onmt.utils.String.pad('', optsz)
    -- second pass to print
    for _,option in ipairs(self.helplines) do
      if type(option) == 'table' then
        io.write('  ')
        if option.default ~= nil then -- it is an option
          io.write(onmt.utils.String.pad(option.key, optsz))
          if option.meta and option.meta.enum then
            io.write(' ('..table.concat(option.meta.enum, ', ')..')')
          end
          option.help = option.help:gsub("\n   *","\n"..padMultiLine.."   ")
          if option.help then io.write(' ' .. option.help) end
          io.write(' [' .. tostring(option.default) .. ']')
        else -- it is an argument
          io.write(onmt.utils.String.pad('<' .. onmt.utils.String.stripHyphens(option.key) .. '>', optsz))
          if option.help then io.write(' ' .. option.help) end
        end
      else
        io.write(option) -- just some additional help
      end
      io.write('\n')
    end
    io.write('\n')
  end
end

function ExtendedCmdLine:option(key, default, help, _meta_)
 parent.option(self, key, default, help)
 self.options[key].meta = _meta_
end

--[[ Override options with option values set in file `filename`. ]]
function ExtendedCmdLine:loadConfig(filename, opt)
  local file = assert(io.open(filename, "r"))

  for line in file:lines() do
    -- Ignore empty or commented out lines.
    if line:len() > 0 and string.sub(line, 1, 1) ~= '#' then
      local field = onmt.utils.String.split(line, '=')
      assert(#field == 2, 'badly formatted config file')

      local key = onmt.utils.String.strip(field[1])
      local val = onmt.utils.String.strip(field[2])

      assert(opt[key] ~= nil, 'unkown option ' .. key)

      opt[key] = convert(key, val, opt[key])
    end
  end

  file:close()
  return opt
end

function ExtendedCmdLine:dumpConfig(opt, filename)
  local file = assert(io.open(filename, 'w'))

  for key, val in pairs(opt) do
    file:write(key .. ' = ' .. tostring(val) .. '\n')
  end

  file:close()
end

function ExtendedCmdLine:parse(arg)
  local i = 1
  local params = self:default()

  local nArgument = 0

  local doHelp = false
  local doMd = false
  local readConfig
  local saveConfig
  while i <= #arg do
    if arg[i] == '-help' or arg[i] == '-h' or arg[i] == '--help' then
      doHelp = true
      i = i + 1
    elseif arg[i] == '-md' then
      doMd = true
      i = i + 1
    elseif arg[i] == '-config' then
      readConfig = arg[i+1]
      i = i + 2
    elseif arg[i] == '-save_config' then
      saveConfig = arg[i+1]
      i = i + 2
    else
      if self.options[arg[i]] then
        i = i + self:__readOption__(params, arg, i)
      else
        nArgument = nArgument + 1
       i = i + self:__readArgument__(params, arg, i, nArgument)
      end
    end
  end

  if doHelp then
    self:help(arg, doMd)
    os.exit(0)
  end

  if nArgument ~= #self.arguments then
    self:error('not enough arguments')
  end

  if readConfig then
    params = self:loadConfig(readConfig, params)
  end

  if saveConfig then
    self:dumpConfig(params, saveConfig)
  end

  for k,v in pairs(params) do
    local K = '-'..k
    if not self.options[K] and self.options[k] then K=k end
    local meta = self.options[K].meta
    if meta then
      if meta.valid and not meta.valid(v) then
        self:error("option '"..k.."' value is not valid")
      end
      if meta.enum and not onmt.utils.Table.hasValue(meta.enum, v) then
        self:error("option '"..k.."' value is not in possible values")
      end
    end
  end

  return params
end

function ExtendedCmdLine:setCmdLineOptions(moduleOptions, group)
  if group then
    self:text("")
    self:text("**"..group.." options**")
    self:text("")
  end
  for i=1,#moduleOptions do
    if type(moduleOptions[i])=='table' then
      self:option(table.unpack(moduleOptions[i]))
    else
      self:argument(moduleOptions[i])
    end
  end
end

function ExtendedCmdLine.getModuleOpts(args, moduleOptions)
  local moduleArgs = {}
  for i=1,#moduleOptions do
    local optname = moduleOptions[i][1]
    if optname:sub(1,1)=='-' then optname = optname:sub(2) end
    moduleArgs[optname] = args[optname]
  end
  return moduleArgs
end

------------------------------------------------------------------------------------------------------------------
-- Validators
------------------------------------------------------------------------------------------------------------------

-- Check if is integer between minValue and maxValue.
function ExtendedCmdLine.isInt(minValue, maxValue)
  return function(v)
    return (math.floor(v) == v and
      (not minValue or v >= minValue) and
      (not maxValue or v <= maxValue))
    end
end

-- Check if is positive integer.
function ExtendedCmdLine.isUInt(maxValue)
  return ExtendedCmdLine.isInt(0, maxValue)
end

-- Check if list of positive integers.
function ExtendedCmdLine.listUInt(v)
  local sv = tostring(v)
  local p = 1
  while true do
    local q
    p, q = sv:find("%d+",p)
    if q == #sv then return true end
    if not p or sv:sub(q+1,q+1) ~= ',' then return false end
    p = q+2
  end
end

-- Check if non empty.
function ExtendedCmdLine.nonEmpty(v)
  return v and v ~= ''
end

-- Check if the corresponding file exists.
function ExtendedCmdLine.fileExists(v)
  return path.exists(v)
end

-- Check non set or if the corresponding file exists.
function ExtendedCmdLine.fileNullOrExists(v)
  return v == '' or ExtendedCmdLine.fileExists(v)
end

return ExtendedCmdLine
