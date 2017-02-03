------------------------------------------------------------------------------------------------------------------
-- Local utility functions
------------------------------------------------------------------------------------------------------------------

local function pad(str, sz)
   return str .. string.rep(' ', sz-#str)
end

local function strip(str)
   return string.match(str, '%-*(.*)')
end

local function _hasValue(list, value)
  for _,v in ipairs(list) do
    if v == value then return true end
  end
  return false
end

------------------------------------------------------------------------------------------------------------------

local extendedCmdLine, parent = torch.class('onmt.extendedCmdLine', 'torch.CmdLine')

--[[
  Extended handling of command line options - provide validation methods, and utilities for handling options
  at module level.

  Example:

    cmd = onmt.utils.ExtendedCmdLine.new()
    local optim_options = {
      {'-max_batch_size',     64   , 'Maximum batch size', {valid=onmt.utils.ExtendedCmdLine.isUInt()}}
    }

    cmd:setCmdLineOptions(optim_options, 'Optimization')
    local opt = cmd:parse(arg)

    local optimArgs = cmd.getModuleOpts(opt, optim_options)
]]

function extendedCmdLine:__init()
  parent.__init(self)
end

function extendedCmdLine:help(arg)
   io.write('Usage: ')
   if arg then io.write(arg[0] .. ' ') end
   io.write('[options] ')
   for i=1,#self.arguments do
      io.write('<' .. strip(self.arguments[i].key) .. '>')
   end
   io.write('\n')

   -- first pass to compute max length
   local optsz = 0
   for _,option in ipairs(self.helplines) do
      if type(option) == 'table' then
         if option.default ~= nil then -- it is an option
            if #option.key > optsz then
               optsz = #option.key
            end
         else -- it is an argument
            if #strip(option.key)+2 > optsz then
               optsz = #strip(option.key)+2
            end
         end
      end
   end

   local padMultiLine = pad('', optsz)
   -- second pass to print
   for _,option in ipairs(self.helplines) do
      if type(option) == 'table' then
         io.write('  ')
         if option.default ~= nil then -- it is an option
            io.write(pad(option.key, optsz))
            if option.meta and option.meta.enum then
              io.write(' ('..table.concat(option.meta.enum, ', ')..')')
            end
            option.help = option.help:gsub("\n   *","\n"..padMultiLine.."   ")
            if option.help then io.write(' ' .. option.help) end
            io.write(' [' .. tostring(option.default) .. ']')
         else -- it is an argument
            io.write(pad('<' .. strip(option.key) .. '>', optsz))
            if option.help then io.write(' ' .. option.help) end
         end
      else
         io.write(option) -- just some additional help
      end
      io.write('\n')
   end
end

function extendedCmdLine:option(key, default, help, _meta_)
  parent.option(self, key, default, help)
  self.options[key].meta = _meta_
end

function extendedCmdLine:parse(arg)
  local params = parent.parse(self, arg)
  for k,v in pairs(params) do
    local meta = self.options['-'..k].meta
    if meta then
      if meta.valid and not meta.valid(v) then
          self:error("option '"..k.."' value is not valid")
      end
      if meta.enum and not _hasValue(meta.enum, v) then
          self:error("option '"..k.."' value is not in possible values")
      end
    end
  end
  return params
end

function extendedCmdLine:setCmdLineOptions(moduleOptions, group)
  if group then
    self:text("")
    self:text("**"..group.." options**")
    self:text("")
  end
  for i=1,#moduleOptions do
    self:option(table.unpack(moduleOptions[i]))
  end
end

function extendedCmdLine.getModuleOpts(args, moduleOptions)
  local moduleArgs = {}
  for i=1,#moduleOptions do
    local optname = moduleOptions[i][1]:sub(2)
    moduleArgs[optname] = args[optname]
  end
  return moduleArgs
end

function extendedCmdLine.isInt(minValue, maxValue)
  return function(v)
    return (math.floor(v) == v and
            (not minValue or v >= minValue) and
            (not maxValue or v <= maxValue))
  end
end

function extendedCmdLine.isUInt(maxValue)
  return extendedCmdLine.isInt(0, maxValue)
end

function extendedCmdLine.nonEmpty(v)
  return v and v ~= ''
end

function extendedCmdLine.fileNullOrExists(v)
  if v == '' then return true end
  return path.exists(v)
end

return extendedCmdLine
