--[[ MemoryOptimizer is a class used for optimizing memory usage of a replicated network.
--]]
local MemoryOptimizer = torch.class('MemoryOptimizer')

-- We cannot share every internal tensors (that is why we need to replicate in the first place).
-- The general rule is to not share tensors whose content is used in the backward pass
-- We allow the sharing when only the size is queried as it is constant during the
-- forward and backward passes.

-- We cannot share the output of these modules as they use it in their backward pass.
local protectOutput = {
  'nn.Sigmoid',
  'nn.SoftMax',
  'nn.Tanh'
}

-- We cannot share the input of these modules as they use it in their backward pass.
local protectInput = {
  'nn.Linear',
  'nn.JoinTable',
  'nn.CMulTable',
  'nn.MM',
  'onmt.Bottle'
}

local function useSameStorage(t, l)
  if torch.isTensor(l) then
    return torch.pointer(t:storage()) == torch.pointer(l:storage())
  elseif torch.type(l) == 'table' then
    for _, m in ipairs(l) do
      if useSameStorage(t, m) then
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
    if not useSameStorage(t, net.gradInput) and not useSameStorage(t, net.output) and not useSameStorage(t, protected) then
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

  for name, mod in pairs(modules) do
    self.modelDesc[name] = {}

    if torch.isTypeOf(mod, 'onmt.Sequencer') then
      -- If the module directly contains a network, take the first clone.
      self.modelDesc[name][1] = {}
      registerNet(self.modelDesc[name][1], mod:net(1), mod.network)
    elseif mod.modules then
      -- Otherwise, look in submodules instead.
      local i = 1
      mod:apply(function(m)
        if torch.isTypeOf(m, 'onmt.Sequencer') then
          self.modelDesc[name][i] = {}
          registerNet(self.modelDesc[name][i], m:net(1), m.network)
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
  for _, desc in pairs(self.modelDesc) do
    for i = 1, #desc do
      local net = desc[i].net
      local base = desc[i].base
      local mempool = {}

      -- Some modules are using output when performing updateGradInput so we cannot share these.
      local protectedOutput = { desc[i].input }
      net:apply(function(m)
        if onmt.utils.Table.hasValue(protectOutput, torch.typename(m)) then
          table.insert(protectedOutput, m.output)
        end
        if onmt.utils.Table.hasValue(protectInput, torch.typename(m)) then
          table.insert(protectedOutput, m.input)
        end
      end)

      local globalIdx = 1
      local idx = 1

      local gradInputMap = {}
      local outputMap = {}

      -- Go over the network to determine which tensors can be shared.
      net:apply(function(m)
        local giSize = getSize(m.gradInput, mempool)
        local oSize = getSize(m.output, mempool)
        totSize = totSize + giSize
        totSize = totSize + oSize
        if canShare(m.gradInput, net, desc[i].gradOutput) then
          sharedSize = sharedSize + giSize
          m.gradInputSharedIdx = idx
          gradInputMap[globalIdx] = idx
          idx = idx + 1
        end
        if canShare(m.output, net, protectedOutput) then
          sharedSize = sharedSize + oSize
          m.outputSharedIdx = idx
          outputMap[globalIdx] = idx
          idx = idx + 1
        end

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
