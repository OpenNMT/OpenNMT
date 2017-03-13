--[[
  MemoryOptimizer is a class used for optimizing memory usage.
  The idea is that each module is using different tensors:
    - params
--]]
local MemoryOptimizer = torch.class('MemoryOptimizer')

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
  'nn.MM'
}

local function contains(list, m)
  for i = 1, #list do
    if torch.typename(m) == list[i] then
      return true
    end
  end
  return false
end

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

-- Returns size and shape of tensor/tensor table
local function getSize(t, mempool)
  local size = 0
  local shape = ''
  if torch.isTensor(t) then
    for i = 1,t:dim() do
      shape = shape .. '/' .. t:size(i)
    end
    if t:storage() then
      if not mempool[torch.pointer(t:storage())] then
        mempool[torch.pointer(t:storage())] = t:storage():size()*t:elementSize()
        return mempool[torch.pointer(t:storage())], shape
      end
    end
  elseif torch.type(t) == 'table' then
    for _, m in ipairs(t) do
      local subSize, subShape = getSize(m, mempool)
      size = size + subSize
      shape = shape .. '-' .. subShape
    end
  end
  return size, shape
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
  memoryOptimizer.optimize(model) -- actual optimization by marking shared tensors

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
      for i = 1, #mod.modules do
        if mod.modules[i].net then
          self.modelDesc[name][i] = {}
          registerNet(self.modelDesc[name][i], mod.modules[i]:net(1), mod.modules[i].network)
        end
      end
    end
  end
end

local function registerStorageDepth(depth, t, depthStorage)
  if type(t) == 'table' then
    for _,v in ipairs(t) do
      registerStorageDepth(depth, v, depthStorage)
    end
  elseif torch.isTensor(t) then
    if not depthStorage[torch.pointer(t:storage())] or depthStorage[torch.pointer(t:storage())] < depth then
      depthStorage[torch.pointer(t:storage())] = depth
    end
  end
end

local function calculateDepths(node, depth, depths, depthStorage)
  if not depths[node.id] or depths[node.id] < depth then
    depths[node.id] = depth
    registerStorageDepth(depth, node.data.gradOutput, depthStorage)
    registerStorageDepth(depth, node.data.input, depthStorage)
    if node.data.module then
      node.data.module:apply(function(m)
        if m.input then
          registerStorageDepth(depth, m.input, depthStorage)
        end
        if m.gradInput then
          registerStorageDepth(depth, m.gradInput, depthStorage)
        end
        if m.output then
          registerStorageDepth(depth, m.output, depthStorage)
        end
      end)
    end
    if node.children then
      for i in ipairs(node.children) do
        calculateDepths(node.children[i], depth+1, depths, depthStorage)
      end
    end
  end
end

local function getMaxDepthStorage(t, depthStorage)
  if type(t) == 'table' then
    local max = -1
    for _,v in ipairs(t) do
      local m = getMaxDepthStorage(v, depthStorage)
      if m > max then
        max = m
      end
    end
    return max
  elseif torch.isTensor(t) then
    if depthStorage[torch.pointer(t:storage())] then
      return depthStorage[torch.pointer(t:storage())]
    else
      return 100000
    end
  end
end

-- a tensor/tensor table shareable between clones can also be recycled "vertically"
-- we check if there is another tensor with exactly the same shape and without any overlap in the
-- calculation graph that we can use
local function getSharedTensor(idx, t, tShape, tSize, depthStorageFwd, depthStorageBwd, MapVShare)
  local fwdDepth = getMaxDepthStorage(t, depthStorageFwd)
  local bwdDepth = getMaxDepthStorage(t, depthStorageBwd)
  local vSave = 0
  local shareIdx
  -- check if we can use another shared index with the same shape/size
  if MapVShare[tShape] then
    for idxMap, ranges in pairs(MapVShare[tShape]) do
      local isOk = true
      for _,v in ipairs(ranges) do
        -- range of usage of the tensor/tensor table can not overlap with our current tensor/tensor table
        if not((fwdDepth>v.fwdDepth and bwdDepth<v.bwdDepth) or
               (fwdDepth<v.fwdDepth and bwdDepth>v.bwdDepth)) then
          isOk = false
          break
        end
      end
      if isOk then
        shareIdx = idxMap
        vSave = tSize
        break
      end
    end
  else
    MapVShare[tShape] = {}
  end
  -- if we cannot recycle a variable, create a new index
  if not shareIdx then
    shareIdx = idx
    idx = idx + 1
  end
  if not MapVShare[tShape][shareIdx] then
    MapVShare[tShape][shareIdx] = {}
  end
  table.insert(MapVShare[tShape][shareIdx], {fwdDepth=fwdDepth, bwdDepth=bwdDepth})
  return idx, shareIdx, vSave
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
  local verticalSizeSave = 0

  for _, desc in pairs(self.modelDesc) do
    for i = 1, #desc do
      local net = desc[i].net
      local base = desc[i].base
      local mempool = {}

      -- calculate depth of the nodes, and storages used in the graph
      -- it will be used to recycle vertically storages
      local depthsFwd = {}
      local depthStorageFwd = {}
      local roots = net.fg:roots()
      for j,_ in ipairs(roots) do
        calculateDepths(roots[j], 0, depthsFwd, depthStorageFwd)
      end
      local depthsBwd = {}
      local depthStorageBwd = {}
      roots = net.bg:roots()
      for j,_ in ipairs(roots) do
        calculateDepths(roots[j], 0, depthsBwd, depthStorageBwd)
      end

      -- Some modules are using output when performing updateGradInput so we cannot share these.
      local protectedOutput = { desc[i].input }
      net:apply(function(m)
        if contains(protectOutput, m) then
          table.insert(protectedOutput, m.output)
        end
        if contains(protectInput, m) then
          table.insert(protectedOutput, m.input)
        end
      end)

      local globalIdx = 1
      local idx = 1

      local gradInputMap = {}
      local MapVShare = {}
      local outputMap = {}

      -- Go over the network to determine which tensors can be shared.
      net:apply(function(m)
        local giSize, giShape = getSize(m.gradInput, mempool)
        local oSize, oShape = getSize(m.output, mempool)
        totSize = totSize + giSize
        totSize = totSize + oSize
        local vSave, shareIdx
        if canShare(m.gradInput, net, desc[i].gradOutput) then
          sharedSize = sharedSize + giSize
          idx, shareIdx, vSave = getSharedTensor(idx, m.gradInput, giShape, giSize, depthStorageFwd, depthStorageBwd, MapVShare)
          verticalSizeSave = verticalSizeSave + vSave
          m.gradInputSharedIdx = shareIdx
          gradInputMap[globalIdx] = shareIdx
        end
        if canShare(m.output, net, protectedOutput) then
          sharedSize = sharedSize + oSize
          idx, shareIdx, vSave = getSharedTensor(idx, m.output, oShape, oSize, depthStorageFwd, depthStorageBwd, MapVShare)
          verticalSizeSave = verticalSizeSave + vSave
          m.outputSharedIdx = shareIdx
          outputMap[globalIdx] = shareIdx
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
  return sharedSize, verticalSizeSave, totSize
end

return MemoryOptimizer
