--[[ Profile is a class used for generating profiling of a training
--]]
local Profiler = torch.class('Profiler')

function Profiler.declareOpts(cmd)
  cmd:option('-profiler', false, [[Generate profiling logs]])
end

--[[ Profiler object

To avoid concurrency problem for parallel processing, each thread should have its own Profiler.
Profiles can be embedded.

Parameters:
  * `opt` - access to program parameter

Example:
    -- global profiler initialization
    globalProfiler = onmt.utils.Profiler.new(opt)

    -- thread-specific profiler
      _G.profiler = onmt.utils.Profiler.new(opt)

      _G.profiler:reset()

      _G.profiler:start("encoder")
      [...]
      _G.profiler:stop("encoder"):start("decoder")
      [...]
      _G.profiler:stop("decoder")

      local profile = _G.profiler.dump()

    -- adds up thread profile
    globalProfiler:add(profile)

    Logger:info(globalProfiler:log())

]]
function Profiler:__init(opt)
  if not opt.profiler then
    self.disable = true
  end
  self:reset()
end

-- Reset Profiler.
function Profiler:reset()
  self.profiles = {}
  self.timers = {}
  self.stack = {}
end

-- Start recording a section.
function Profiler:start(name)
  if self.disable then return self end
  self.timers[name] = torch.Timer()
  table.insert(self.stack, name)
  return self
end

-- Stop recording a section.
function Profiler:stop(name)
  if self.disable then return self end
  assert(self.stack[#self.stack] == name, 'Invalid profiler stop action: '..name)
  local path = table.concat(self.stack, ".")
  if not self.profiles[path] then self.profiles[path] = 0 end
  self.profiles[path] = self.profiles[path] + self.timers[name]:time().real
  self.timers[name] = nil
  table.remove(self.stack)
  return self
end

-- Dump profile.
function Profiler:dump()
  if self.disable then return end
  return self.profiles
end

-- Aggregage profiles with a previous dump.
function Profiler:add(profile)
  if self.disable then return end
  for name,v in pairs(profile) do
    if not self.profiles[name] then self.profiles[name] = 0 end
    self.profiles[name] = self.profiles[name] + v
  end
end

-- Returns text string with log structured by sub level.
-- train:[23,encoder_fwd:10,encoder_bwd:14]
function Profiler:log(prefix)
  prefix = prefix or ''
  local t = {}
  for name,v in pairs(self.profiles) do
    v = string.format("%g", v)
    if name:sub(1,#prefix) == prefix then
      local pos = #prefix + 1
      if not name:sub(pos):find("%.") then
        local npref = prefix
        local subtree = self:log(name..'.')
        if #subtree > 0 then
          v='['..v..', '..subtree..']'
        end
        table.insert(t, name:sub(pos)..':'..v)
      end
    end
  end
  return table.concat(t, ", ")
end

return Profiler
