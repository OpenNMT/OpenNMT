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
function HookManager:__init(args)
  self.hooks = {}
  if args.hook_file ~= '' then
    local _, err = pcall(function()
      local hooks = require(args.hook_file)
      assert(type(hooks) == 'table')
      for n, v in pairs(hooks) do
        assert(type(n) == 'string')
        assert(type(v) == 'function')
        self.hooks[n] = v
        _G.logger:info("Register hook '"..n.."' from "..args.hook_file)
      end
    end)
    onmt.utils.Error.assert(not err, 'Cannot load hooks ('..args.hook_file..') - %s', err)
  end
end

function HookManager:call(method, ...)
  if self.hooks[method] then
    return self.hooks[method](...)
  end
end

return HookManager
