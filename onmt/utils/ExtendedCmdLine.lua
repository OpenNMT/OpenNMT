local extendedCmdLine, parent = torch.class('onmt.extendedCmdLine', 'torch.CmdLine')
local path = require('pl.path')

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

function extendedCmdLine:help(arg, doMd)
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
  else
   io.write('Usage: ')
   if arg then io.write(arg[0] .. ' ') end
   io.write('[options] ')
   for i=1,#self.arguments do
      io.write('<' .. onmt.utils.String.stripHyphens(self.arguments[i].key) .. '>')
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
  end
end

function extendedCmdLine:option(key, default, help, _meta_)
  parent.option(self, key, default, help)
  self.options[key].meta = _meta_
end

function extendedCmdLine:parse(arg)
   local i = 1
   local params = self:default()

   local nArgument = 0

   local doHelp = false
   local doMd = false
   while i <= #arg do
      if arg[i] == '-help' or arg[i] == '-h' or arg[i] == '--help' then
        doHelp = true
        i = i + 1
      elseif arg[i] == '-md' then
        doMd = true
        i = i + 1
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

  for k,v in pairs(params) do
    local meta = self.options['-'..k].meta
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
