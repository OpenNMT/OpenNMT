---------------------------------------------------------------------------------
-- Local utility functions
---------------------------------------------------------------------------------

local function wrapIndent(text, size, pad)
  local p = 0
  while true do
    local q = text:find(" ", size+p)
    if not q then
      return text
    end
    text = text:sub(1, q) .. "\n" ..pad .. text:sub(q+1, #text)
    p = q+2+#pad
  end
end

---------------------------------------------------------------------------------

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
    local options = {
      {'-max_batch_size',     64   , 'Maximum batch size', {valid=onmt.utils.ExtendedCmdLine.isUInt()}}
    }

    cmd:setCmdLineOptions(options, 'Optimization')
    local opt = cmd:parse(arg)

    local optimArgs = cmd.getModuleOpts(opt, options)

  Additional meta-fields:

  * `valid`: validation method
  * `enum`: enumeration list
  * `structural`: if defined, mark a structural parameter - 0 means cannot change value, 1 means that it can change dynamically
  * `init_only`: if true, mark a parameter that can only be set at init time
  * `train_state`: if true, the option will be automatically reused when continuing a training

]]

function ExtendedCmdLine:__init(script)
  self.script = script
  parent.__init(self)

  self:text('')
  self:option('-h', false, 'This help.')
  self:option('-md', false, 'Dump help in Markdown format.')
  self:option('-config', '', 'Load options from this file.', {valid=ExtendedCmdLine.fileNullOrExists})
  self:option('-save_config', '', 'Save options to this file.')
end

function ExtendedCmdLine:help(arg, doMd)
  if doMd then
    io.write('`' .. self.script .. '` options:\n')
    for _, option in ipairs(self.helplines) do
      if type(option) == 'table' then
        io.write('* ')
        if option.default ~= nil then -- It is an option.
          local optType = '<' .. option.type .. '>'
          if option.type == 'boolean' then
            optType = '[' .. optType .. ']'
          end
          io.write('`' .. option.key .. ' ' .. optType .. '`')

          local valInfo = {}
          if option.meta and option.meta.required then
            table.insert(valInfo, 'required')
          end
          if option.meta and option.meta.enum then
            for k, v in pairs(option.meta.enum) do
              option.meta.enum[k] = '`' .. v .. '`'
            end
            table.insert(valInfo, 'accepted: ' .. table.concat(option.meta.enum, ', '))
          end
          local default
          if type(option.default) == 'table' then
            default = table.concat(option.default, ', ')
          elseif option.default == '' then
            default = '\'\''
          else
            default = tostring(option.default)
          end
          if not(option.meta and option.meta.required and default == '\'\'') then
            table.insert(valInfo, 'default: `' .. default .. '`')
          end
          if #valInfo > 0 then
            io.write(' (' .. table.concat(valInfo, '; ') .. ')')
          end

          io.write('<br/>')

          option.help = option.help:gsub(' *\n   *', ' ')
          if option.help then
            io.write(option.help)
          end
        else -- It is an argument.
          io.write('<' .. onmt.utils.String.stripHyphens(option.key) .. '>')
          if option.help then
            io.write(' ' .. option.help)
          end
        end
      else
        local display = option:gsub('%*', '')
        if display:len() > 0 then
          io.write('## ')
        end
        io.write(display) -- Just some additional help.
      end
      io.write('\n')
    end
    io.write('\n')
  else
    if arg then
      io.write('Usage: ')
      io.write(arg[0] .. ' ' .. self.script .. ' ')
      io.write('[options] ')
      for i = 1, #self.arguments do
        io.write('<' .. onmt.utils.String.stripHyphens(self.arguments[i].key) .. '>')
      end
      io.write('\n')
    end

    -- First pass to compute max length.
    local optsz = 0
    for _, option in ipairs(self.helplines) do
      if type(option) == 'table' then
        if option.default ~= nil then -- It is an option.
          if #option.key > optsz then
            optsz = #option.key
          end
        else -- It is an argument.
          local stripOptionKey = onmt.utils.String.stripHyphens(option.key)
          if #stripOptionKey + 2 > optsz then
            optsz = #stripOptionKey + 2
          end
        end
      end
    end

    local padMultiLine = onmt.utils.String.pad('', optsz)
    -- Second pass to print.
    for _, option in ipairs(self.helplines) do
      if type(option) == 'table' then
        io.write('  ')
        if option.default ~= nil then -- It is an option.
          io.write(onmt.utils.String.pad(option.key, optsz))
          local msg = ''
          msg = msg .. option.help:gsub('\n', ' ')
          local valInfo = {}
          if option.meta and option.meta.required then
            table.insert(valInfo, 'required')
          end
          if option.meta and option.meta.enum then
            table.insert(valInfo, 'accepted: ' .. table.concat(option.meta.enum, ', '))
          end
          local default
          if type(option.default) == 'table' then
            default = table.concat(option.default, ', ')
          elseif option.default == '' then
            default = '\'\''
          else
            default = tostring(option.default)
          end
          if not(option.meta and option.meta.required and default == '\'\'') then
            table.insert(valInfo, 'default: `' .. default .. '`')
          end
          if #valInfo > 0 then
            msg = msg .. ' (' .. table.concat(valInfo, '; ') .. ')'
          end
          io.write(' ' .. wrapIndent(msg:gsub('  *', ' '),60,padMultiLine..'     '))
        else -- It is an argument.
          io.write(onmt.utils.String.pad('<' .. onmt.utils.String.stripHyphens(option.key) .. '>', optsz))
          if option.help then
            io.write(' ' .. option.help)
          end
        end
      else
        io.write(option) -- Just some additional help.
      end
      io.write('\n')
    end
    io.write('\n')
  end
end

function ExtendedCmdLine:error(msg)
  if self.script then
    io.stderr:write(self.script .. ': ')
  end
  io.stderr:write(msg .. '\n')
  io.stderr:write('Try \'' .. self.script .. ' -h\' for more information, or visit the online documentation at http://opennmt.net/OpenNMT/.\n')
  os.exit(1)
end

function ExtendedCmdLine:option(key, default, help, _meta_)
  for _,v in ipairs(self.helplines) do
    if v.key == key then
      return
    end
  end
  parent.option(self, key, default, help)

  -- check if option correctly defined - if default value does not match validation criterion then it is either
  -- empty and in that case, is a required option, or is an error
  if _meta_ and (
    (_meta_.valid and not _meta_.valid(default)) or
    (_meta_.enum and not onmt.utils.Table.hasValue(_meta_.enum, default))) then
    assert(default=='',"Invalid option default definition: "..key.."="..default)
    _meta_.required = true
  end

  self.options[key].meta = _meta_
end

--[[ Override options with option values set in file `filename`. ]]
function ExtendedCmdLine:loadConfig(filename, opt)
  local file = assert(io.open(filename, 'r'))

  for line in file:lines() do
    -- Ignore empty or commented out lines.
    if line:len() > 0 and string.sub(line, 1, 1) ~= '#' then
      local field = onmt.utils.String.split(line, '=')
      assert(#field == 2, 'badly formatted config file')

      local key = onmt.utils.String.strip(field[1])
      local val = onmt.utils.String.strip(field[2])

      -- Rely on the command line parser.
      local arg = { '-' .. key }

      if val == '' then
        table.insert(arg, '')
      else
        onmt.utils.Table.append(arg, onmt.utils.String.split(val, ' '))
      end

      self:__readOption__(opt, arg, 1)
      opt._is_default[key] = nil
    end
  end

  file:close()
  return opt
end

function ExtendedCmdLine:logConfig(opt)
  local keys = {}
  for key in pairs(opt) do
    table.insert(keys, key)
  end

  table.sort(keys)
  _G.logger:debug('Options:')

  for _, key in ipairs(keys) do
    if key:sub(1, 1) ~= '_' then
      local val = opt[key]
      if type(val) == 'string' then
        val = '\'' .. val .. '\''
      elseif type(val) == 'table' then
        val = table.concat(val, ' ')
      else
        val = tostring(val)
      end
      _G.logger:debug(' * ' .. key .. ' = ' .. tostring(val))
    end
  end
end

function ExtendedCmdLine:dumpConfig(opt, filename)
  local file = assert(io.open(filename, 'w'))

  for key, val in pairs(opt) do
    if key:sub(1, 1) ~= '_' then
      if type(val) == 'table' then
        val = table.concat(val, ' ')
      else
        val = tostring(val)
      end
      file:write(key .. ' = ' .. val .. '\n')
    end
  end

  file:close()
end

--[[ Convert `val` string to the target type. ]]
function ExtendedCmdLine:convert(key, val, type, subtype)
  if not type or type == 'string' then
    val = val
  elseif type == 'table' then
    local values = {}
    val = onmt.utils.String.split(val, ' ')
    for _, v in ipairs(val) do
      onmt.utils.Table.append(values, onmt.utils.String.split(v, ','))
    end
    for i = 1, #values do
      values[i] = self:convert(key, values[i], subtype)
    end
    val = values
  elseif type == 'number' then
    val = tonumber(val)
  elseif type == 'boolean' then
    if val == '0' or val == 'false' then
      val = false
    elseif val == '1' or val == 'true' then
      val = true
    else
      self:error('invalid argument for boolean option ' .. key .. ' (should be 0, 1, false or true)')
    end
  else
    self:error('unknown required option type ' .. type)
  end

  if val == nil then
    self:error('invalid type for option ' .. key .. ' (should be ' .. type .. ')')
  end

  return val
end

function ExtendedCmdLine:__readOption__(params, arg, i)
  local key = arg[i]
  local option = self.options[key]
  if not option then
    self:error('unknown option ' .. key)
  end

  local multiValues = (option.type == 'table')
  local argumentType
  if not multiValues then
    argumentType = option.type
  elseif #option.default > 0 then
    argumentType = type(option.default[1])
  end

  local values = {}
  local numArguments = 0

  -- browse through parameters till next potential option (starting with -Letter)
  while arg[i + 1] and string.find(arg[i+1],'-%a')~=1 do
    local value = self:convert(key, arg[i + 1], option.type, argumentType)

    if type(value) == 'table' then
      onmt.utils.Table.append(values, value)
    else
      table.insert(values, value)
    end

    i = i + 1
    numArguments = numArguments + 1
  end

  local optionName = onmt.utils.String.stripHyphens(key)

  if #values == 0 then
    if argumentType == 'boolean' then
      params[optionName] = not option.default
    else
      self:error('missing argument(s) for option ' .. key)
    end
  elseif multiValues then
    params[optionName] = values
  elseif #values > 1 then
    self:error('option ' .. key  .. ' expects 1 argument but ' .. #values .. ' were given')
  else
    params[optionName] = values[1]
  end

  return numArguments + 1
end

function ExtendedCmdLine:parse(arg)
  local i = 1

  -- set default value
  local params = { _is_default={}, _structural={}, _init_only={}, _train_state={} }
  for option,v in pairs(self.options) do
    local soption = onmt.utils.String.stripHyphens(option)
    params[soption] = v.default
    params._is_default[soption] = true
  end

  local nArgument = 0

  local doHelp = false
  local doMd = false
  local readConfig
  local saveConfig

  local cmdlineOptions = {}

  while i <= #arg do
    if arg[i] == '-help' or arg[i] == '-h' or arg[i] == '--help' then
      doHelp = true
      i = i + 1
    elseif arg[i] == '-md' then
      doMd = true
      i = i + 1
    elseif arg[i] == '-config' then
      readConfig = arg[i + 1]
      i = i + 2
    elseif arg[i] == '-save_config' then
      saveConfig = arg[i + 1]
      i = i + 2
    else
      local sopt = onmt.utils.String.stripHyphens(arg[i])
      params._is_default[sopt] = nil
      if self.options[arg[i]] then
        if cmdlineOptions[arg[i]] then
          self:error('duplicate cmdline option: '..arg[i])
        end
        cmdlineOptions[arg[i]] = true
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

  for k, v in pairs(params) do
    if k:sub(1, 1) ~= '_' then
      local K = '-' .. k
      if not self.options[K] and self.options[k] then
        K = k
      end
      local meta = self.options[K].meta
      if meta then
        -- check option validity
        local isValid = true
        local reason = nil

        if meta.depends then
          isValid, reason = meta.depends(params)
          if not isValid then
            local msg = 'invalid dependency for option -'..k
            if reason then
              msg = msg .. ': ' .. reason
            end
            self:error(msg)
          end
        end

        if meta.valid then
          isValid, reason = meta.valid(v)
        end

        if not isValid then
          local msg = 'invalid argument for option -' .. k
          if reason then
            msg = msg .. ': ' .. reason
          end
          self:error(msg)
        end

        if meta.enum and not onmt.utils.Table.hasValue(meta.enum, v) then
          self:error('option -' .. k.. ' is not in accepted values: ' .. table.concat(meta.enum, ', '))
        end
        if meta.structural then
          params._structural[k] = meta.structural
        end
        if meta.init_only then
          params._init_only[k] = meta.init_only
        end
        if meta.train_state then
          params._train_state[k] = meta.train_state
        end
      end
    end
  end

  return params
end

function ExtendedCmdLine:setCmdLineOptions(moduleOptions, group)
  if group and group ~= self.prevGroup then
    self:text('')
    self:text('**' .. group .. ' options**')
    self:text('')
    self.prevGroup = group
  end

  for i = 1, #moduleOptions do
    if type(moduleOptions[i]) == 'table' then
      self:option(table.unpack(moduleOptions[i]))
    else
      self:argument(moduleOptions[i])
    end
  end
end

function ExtendedCmdLine.getModuleOpts(args, moduleOptions)
  local moduleArgs = {}
  for i = 1, #moduleOptions do
    local optname = moduleOptions[i][1]
    if optname:sub(1, 1) == '-' then
      optname = optname:sub(2)
    end
    moduleArgs[optname] = args[optname]
  end
  return moduleArgs
end

function ExtendedCmdLine.getArgument(args, optName)
  for i = 1, #args do
    if args[i] == optName and i < #args then
      return args[i + 1]
    end
  end
  return nil
end

---------------------------------------------------------------------------------
-- Validators
---------------------------------------------------------------------------------

local function buildRangeError(prefix, minValue, maxValue)
  local err = 'the ' .. prefix .. ' must be'
  if minValue then
    err = err .. ' greater than ' .. minValue
  end
  if maxValue then
    if minValue then
      err = err .. ' and'
    end
    err = err .. ' lower than ' .. maxValue
  end
  return err
end

-- Check if is integer between minValue and maxValue.
function ExtendedCmdLine.isInt(minValue, maxValue)
  return function(v)
    return (math.floor(v) == v and
      (not minValue or v >= minValue) and
      (not maxValue or v <= maxValue)),
      buildRangeError('integer', minValue, maxValue)
    end
end

-- Check if is positive integer.
function ExtendedCmdLine.isUInt(maxValue)
  return ExtendedCmdLine.isInt(0, maxValue)
end

-- Check if value between minValue and maxValue.
function ExtendedCmdLine.isFloat(minValue, maxValue)
  return function(v)
    return (type(v) == 'number' and
      (not minValue or v >= minValue) and
      (not maxValue or v <= maxValue)),
      buildRangeError('number', minValue, maxValue)
    end
end

-- Check if non empty.
function ExtendedCmdLine.nonEmpty(v)
  return v and v ~= '', 'the argument must not be empty'
end

-- Check if the corresponding file exists.
function ExtendedCmdLine.fileExists(v)
  return path.exists(v), 'the file must exist'
end

-- Check non set or if the corresponding file exists.
function ExtendedCmdLine.fileNullOrExists(v)
  return v == '' or ExtendedCmdLine.fileExists(v), 'if set, the file must exist'
end

-- Check it is a directory and some file exists
function ExtendedCmdLine.dirStructure(files)
  return function(v)
    for _,f in ipairs(files) do
      if not path.exists(v.."/"..f) then
        return false, 'the directory must exist'
      end
    end
    return true
  end
end

return ExtendedCmdLine
