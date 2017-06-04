--[[ MemoryOptimizer is a class used for optimizing memory usage
--]]
local MemoryOptimizer = torch.class('MemoryOptimizer')

local function tensorIncluded(t, l)
  if torch.isTensor(l) then
    return torch.pointer(t:storage()) == torch.pointer(l:storage())
  elseif torch.type(l) == 'table' then
    for _, m in ipairs(l) do
      if tensorIncluded(t, m) then
        return true
      end
    end
  end
  return false
end

-- We cannot share a tensor if it is exposed or coming from outside of the net
-- otherwise we could generate side-effects.
local function canShare(t, net, protected)
  if torch.isTensor(t) and t:storage() then
    if not tensorIncluded(t, net.gradInput) and not tensorIncluded(t, net.output) and not tensorIncluded(t, protected) then
      return true
    end
  elseif torch.type(t) == 'table' then
    for _, m in ipairs(t) do
      if not canShare(m, net, protected) then
        return false
      end
    end
    return true
  end
  return false
end

local function getSize(t, mempool)
  local size = 0
  if torch.isTensor(t) then
    if t:storage() then
      if not mempool[torch.pointer(t:storage())] then
        mempool[torch.pointer(t:storage())] = t:storage():size()*t:elementSize()
        return mempool[torch.pointer(t:storage())]
      end
    end
  elseif torch.type(t) == 'table' then
    for _, m in ipairs(t) do
      size = size + getSize(m, mempool)
    end
  end
  return size
end

-- Convenience function to register a network to optimize.
local function registerNet(store, net, base)
  store.net = net
  store.base = base
  store.forward = net.forward
  net.forward = function(network, input)
    store.input = input
    return store.forward(network, input)
  end
  store.backward = net.backward
  net.backward = function(network, input, gradOutput)
    store.gradOutput = gradOutput
    return store.backward(network, input, gradOutput)
  end

  -- Add a wrapper around updateOutput to catch the module input.
  net:apply(function (m)
    local updateOutput = m.updateOutput
    m.updateOutput = function (mod, input)
      mod.input = input
      return updateOutput(mod, input)
    end
  end)
end

--[[ Construct a MemoryOptimizer object. In this function, forward and backward function will
--  be overwrited to record input and gradOutput in order to determine which tensors can be shared.

Parameters:
  * `modules` - a list of modules to optimize.

Example:

  local memoryOptimizer = onmt.utils.MemoryOptimizer.new(model) -- prepare memory optimization.
  model:forward(...) -- initialize output tensors
  model:backward(...) -- intialize gradInput tensors
  memoryOptimizer.optimize() -- actual optimization by marking shared tensors

]]
function MemoryOptimizer:__init(modules)
  self.modelDesc = {}

  self.MG=onmt.utils.MemoryGraph.new()

  for name, mod in pairs(modules) do
    self.modelDesc[name] = {}

    if torch.isTypeOf(mod, 'onmt.Sequencer') then
      -- If the module directly contains a network, take the first clone.
      self.modelDesc[name][1] = {}
      registerNet(self.modelDesc[name][1], mod:net(1), mod.network)
      self.MG:add(name, mod:net(1))
    elseif mod.modules then
      -- Otherwise, look in submodules instead.
      local i = 1
      mod:apply(function(m)
        if torch.isTypeOf(m, 'onmt.Sequencer') then
          self.modelDesc[name][i] = {}
          registerNet(self.modelDesc[name][i], m:net(1), m.network)
          self.MG:add(name, m:net(1))
          i = i + 1
        end
      end)
    end
  end

  if onmt.utils.Table.empty(self.modelDesc) then
    _G.logger:warning('Only networks inheriting from onmt.Sequencer can be optimized')
  end
end

--[[ Enable memory optimization by marking tensors to share. Note that the modules must have been initialized
-- by calling forward() and backward() before calling this function and after calling the MemoryOptimizer constructor.

Returns:
  1. `sharedSize` - shared tensor size
  2. `totSize` - total tensor size
]]
function MemoryOptimizer:optimize()
  local totSize = 0
  local sharedSize = 0
  if #self.MG.graphs > 0 then
    -- using memgraph module to explore computation graph and improve memory sharing
    local protectedSize
    totSize, protectedSize = self.MG:optimize()
    sharedSize = totSize - protectedSize
  else
    for _, desc in pairs(self.modelDesc) do
      for i = 1, #desc do
        local net = desc[i].net
        local mempool = {}

        -- Some modules are using output when performing updateGradInput so we cannot share these.
        local protectedInputBuffer = { desc[i].input }
        net:apply(function(m)
          if onmt.utils.MemoryGraph.protected(m, "output") then
            table.insert(protectedInputBuffer, m.output)
          end
          if onmt.utils.MemoryGraph.protected(m, "input") then
            table.insert(protectedInputBuffer, m.input)
          end
        end)

        local idx = 1

        -- Go over the network to determine which tensors can be shared.
        net:apply(function(m)
          local giSize = getSize(m.gradInput, mempool)
          local oSize = getSize(m.output, mempool)
          totSize = totSize + giSize
          totSize = totSize + oSize

          if canShare(m.gradInput, net, desc[i].gradOutput) then
            sharedSize = sharedSize + giSize
            m.gradInputSharedIdx = idx
            idx = idx + 1
          end
          if canShare(m.output, net, protectedInputBuffer) then
            sharedSize = sharedSize + oSize
            m.outputSharedIdx = idx
            idx = idx + 1
          end

          -- Remove the wrapper around updateOutput to catch the module input.
          m.updateOutput = nil
          m.input = nil
        end)

      end
    end
  end

  for _, desc in pairs(self.modelDesc) do
    for i = 1, #desc do
      local net = desc[i].net
      local base = desc[i].base

      local gradInputMap = {}
      local outputMap = {}

      local globalIdx = 1

      -- Mark shareable tensors in the base network.
      net:apply(function(m)
        gradInputMap[globalIdx] = m.gradInputSharedIdx
        outputMap[globalIdx] = m.outputSharedIdx

        -- Remove the wrapper around updateOutput to catch the module input.
        m.updateOutput = nil
        m.input = nil

        globalIdx = globalIdx + 1
      end)

      globalIdx = 1

      -- Mark shareable tensors in the base network.
      base:apply(function (m)
        if gradInputMap[globalIdx] then
          m.gradInputSharedIdx = gradInputMap[globalIdx]
        end
        if outputMap[globalIdx] then
          m.outputSharedIdx = outputMap[globalIdx]
        end
        globalIdx = globalIdx + 1
      end)

      -- Restore function on network backward/forward interception input.
      net.backward = nil
      net.forward = nil
    end
  end

  return sharedSize, totSize
end

return MemoryOptimizer
