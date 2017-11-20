--[[ HookManager is a class for handling "hooks" at several place in the code for introducing programmatically specific processing
--]]
local HookManager = torch.class('HookManager')

local options = {
  {
    '-hook_file', '',
    [[Pointer to a lua file registering hooks for the current process]]
  }
}

function HookManager.declareOpts(cmd)
  cmd:setCmdLineOptions(options, 'HookManager')
end

--[[ Construct a HookManager object.

Parameters:
  * `args` - commandline args, including  HookManager specific options
]]
function HookManager:__init(args, context)
  _G.luaopen_context = context
  self.hooks = {}
  local hook_file = (type(args) == "string" and args) or
                    (type(args) == "table" and args.hook_file)
  if hook_file and hook_file ~= '' then
    local _, err = pcall(function()
      local hooks = require(hook_file)
      assert(type(hooks) == 'table')
      for n, v in pairs(hooks) do
        assert(type(n) == 'string')
        assert(type(v) == 'function')
        self.hooks[n] = v
            if _G.logger then
          _G.logger:info("Register hook '"..n.."' from "..hook_file)
        end
      end
    end)
    if _G.logger then
      onmt.utils.Error.assert(not err, 'Cannot load hooks ('..hook_file..') - %s', err)
    else
      assert(not err, 'Cannot load hooks ('..hook_file..'): '..(err or ""))
    end
  end
end

function HookManager.updateOpt(arg, cmd)
  for i, k in ipairs(arg) do
    if k == '-hook_file' then
      local hookManager = HookManager.new({hook_file=arg[i+1]})
      if hookManager.hooks["declareOpts"] then
        local hook_cmd = onmt.utils.ExtendedCmdLine.new('hook options')
        hookManager:call("declareOpts", hook_cmd)
        cmd:merge(hook_cmd)
      end
    end
  end
end

function HookManager:call(method, ...)
  if self.hooks[method] then
    return self.hooks[method](...)
  end
end

return HookManager
