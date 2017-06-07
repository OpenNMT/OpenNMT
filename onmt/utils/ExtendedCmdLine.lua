---------------------------------------------------------------------------------
-- Local utility functions
---------------------------------------------------------------------------------

local function wrapIndent(text, size, pad)
  text = pad .. text
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

local function concatValues(t)
  local isBoolean
  local s = ''
  for _, v in ipairs(t) do
    if type(v) == 'boolean' then
      isBoolean = true
    else
      if s ~= '' then
        s = s .. ', '
      end
      s = s .. v
    end
  end
  if isBoolean then
    s = 'true, false, ' .. s
  end
  return s
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

function ExtendedCmdLine:help(arg, md)
  if arg then
    if md then
      io.write('`' .. self.script .. '` options:\n')
    else
      io.write('Usage: ')
      io.write(arg[0] .. ' ' .. self.script .. ' ')
      io.write('[options] ')
      for i = 1, #self.arguments do
        io.write('<' .. onmt.utils.String.stripHyphens(self.arguments[i].key) .. '>')
      end
      io.write('\n')
    end
  end

  for _, option in ipairs(self.helplines) do
    if type(option) ~= 'table' then
      if md and option:len() > 0 then
        io.write('## ')
      end
      io.write(option)
      io.write('\n')
    elseif not option.meta or not option.meta.deprecatedBy then
      -- Argument type.
      local argType = '<' .. option.type .. '>'
      if option.type == 'boolean' then
        if option.meta and option.meta.enum then
          argType = argType .. '/<string>'
        end
        argType = '[' .. argType .. ']'
      end

      local argMeta = {}

      -- Argument constraints.
      if option.meta and not option.meta.argument and option.meta.required then
        table.insert(argMeta, 'required')
      end
      if option.meta and option.meta.enum then
        if md then
          for k, v in pairs(option.meta.enum) do
            option.meta.enum[k] = '`' .. tostring(v) .. '`'
          end
        end
        table.insert(argMeta, 'accepted: ' .. concatValues(option.meta.enum))
      end

      if not option.meta or not option.meta.argument then
        -- Default value.
        local argDefault
        if type(option.default) == 'table' then
          argDefault = table.concat(option.default, ', ')
        elseif option.default == '' then
          argDefault = '\'\''
        else
          argDefault = tostring(option.default)
        end
        if not (option.meta and option.meta.required and argDefault == '\'\'') and
           not (type(option.default == 'table') and argDefault == '') then
          if md then
            argDefault = '`' .. argDefault .. '`'
          end
          table.insert(argMeta, 'default: ' .. argDefault)
        end
      end

      local optionPattern = option.key .. ' ' .. argType

      if option.meta and option.meta.argument then
        optionPattern = '<' .. option.key .. '>'
      end

      if md then
        io.write('* `' .. optionPattern.. '`')
      else
        io.write('  ' .. optionPattern)
      end

      if #argMeta > 0 then
        io.write(' ('.. table.concat(argMeta, '; ') .. ')')
      end

      local description = string.gsub(option.help, ' *\n   *', ' ')

      if md then
        io.write('<br/>')
        io.write(description)
      else
        io.write('\n')
        io.write(wrapIndent(description, 60, '      '))
        io.write('\n')
      end

      io.write('\n')
    end
  end
end

function ExtendedCmdLine:error(msg)
  if not self.script then
    error(msg)
  else
    io.stderr:write(self.script .. ': ' .. msg .. '\n')
    io.stderr:write('Try \'' .. self.script .. ' -h\' for more information, or visit the online documentation at http://opennmt.net/OpenNMT/.\n')
    os.exit(1)
  end
end

function ExtendedCmdLine:argument(key, type, help, _meta_)
  for _,v in ipairs(self.helplines) do
    if v.key == key then
      return
    end
  end

  assert(not(self.options[key]) and not(self.arguments[key]), "Duplicate options/arguments: "..key)

  parent.argument(self, key, help, type)

  if not _meta_ then
    _meta_ = {}
  end
  _meta_.argument = true
  self.arguments[#self.arguments].meta = _meta_
  self.options[key] = { meta=_meta_}
end

function ExtendedCmdLine:option(key, default, help, _meta_)
  for _,v in ipairs(self.helplines) do
    if v.key == key then
      return
    end
  end

  assert(not(self.options[key]) and not(self.arguments[key]), "Duplicate options/arguments: "..key)

  parent.option(self, key, default, help)

  -- check if option correctly defined - if default value does not match validation criterion then it is either
  -- empty and in that case, is a required option, or is an error
  if _meta_ and (
    (_meta_.valid and not _meta_.valid(default)) or
    (_meta_.enum and type(default) ~= 'table' and not onmt.utils.Table.hasValue(_meta_.enum, default))) then
    assert(default=='',"Invalid option default definition: "..key.."="..default)
    _meta_.required = true
  end

  if _meta_ and _meta_.enum and type(default) == 'table' then
    for _,k in ipairs(default) do
      assert(onmt.utils.Table.hasValue(_meta_.enum, k), "table option not compatible with enum: "..key)
    end
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
function ExtendedCmdLine:convert(key, val, type, subtype, meta)
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
      -- boolean option can take 3rd values
      if not (meta and meta.enum) then
        self:error('invalid argument for boolean option ' .. key .. ' (should be 0, 1, false or true)')
      end
    end
  else
    self:error('unknown required option type ' .. type)
  end

  if val == nil then
    self:error('invalid type for option ' .. key .. ' (should be ' .. type .. ')')
  end

  return val
end

function ExtendedCmdLine:__readArgument__(params, arg, i, nArgument)
   local argument = self.arguments[nArgument]
   local value = arg[i]

   if nArgument > #self.arguments then
      self:error('invalid argument: ' .. value)
   end
   if argument.type and type(value) ~= argument.type then
      self:error('invalid argument type for argument ' .. argument.key .. ' (should be ' .. argument.type .. ')')
   end

   params[argument.key] = value
   return 1
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
    local value = self:convert(key, arg[i + 1], option.type, argumentType, option.meta)

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
      if arg[i]:sub(1,1) == '-' then
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
    self:error('not enough arguments ')
  end

  if readConfig then
    params = self:loadConfig(readConfig, params)
  end

  if saveConfig then
    self:dumpConfig(params, saveConfig)
  end

  for k, v in pairs(params) do
    if k:sub(1, 1) ~= '_' then
      local K = k
      if not self.options[k] and self.options['-' .. k] then
        K = '-' .. k
      end
      local meta = self.options[K].meta
      if meta then
        -- check option validity
        local isValid = true
        local reason = nil

        if not params._is_default[k] and meta.deprecatedBy then
          local newOption = meta.deprecatedBy[1]
          local newValue = meta.deprecatedBy[2]
          io.stderr:write('DEPRECATION WARNING: option \'-' .. k .. '\' is replaced by \'-' .. newOption .. ' ' .. newValue .. '\'.\n')
          params[newOption] = newValue
        end

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

        if meta.enum and type(self.options[K].default) ~= 'table' and not onmt.utils.Table.hasValue(meta.enum, v) then
          self:error('option -' .. k.. ' is not in accepted values: ' .. concatValues(meta.enum))
        end
        if meta.enum and type(self.options[K].default) == 'table' then
          for _, v1 in ipairs(v) do
            if not onmt.utils.Table.hasValue(meta.enum, v1) then
              self:error('option -' .. k.. ' is not in accepted values: ' .. concatValues(meta.enum))
            end
          end
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
    self:text(group .. ' options')
    self:text('')
    self.prevGroup = group
  end

  for i = 1, #moduleOptions do
    if moduleOptions[i][1]:sub(1,1) == '-' then
      self:option(table.unpack(moduleOptions[i]))
    else
      self:argument(table.unpack(moduleOptions[i]))
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
