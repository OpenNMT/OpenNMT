local ExtendedCmdLine = require('onmt.utils.ExtendedCmdLine')

local Cuda = {
  fp16 = false,
  gpuIds = {},
  deviceIds = {},
  cpuTraining = false
}

local options = {
  {
    '-gpuid', '0',
    [[List of comma-separated GPU identifiers (1-indexed). CPU is used when set to 0.]],
    {
      valid = ExtendedCmdLine.listUInt
    }
  },
  {
    '-fallback_to_cpu', false,
    [[If GPU can't be used, rollback on the CPU.]]
  },
  {
    '-fp16', false,
    [[Use half-precision float on GPU.]]
  },
  {
    '-no_nccl', false,
    [[Disable usage of nccl in parallel mode.]]
  }
}

function Cuda.declareOpts(cmd)
  cmd:setCmdLineOptions(options, 'Cuda')
end

function Cuda.init(opt, deviceId)
  if not deviceId then
    local usedGpuIds = {}
    for _, val in ipairs(onmt.utils.String.split(opt.gpuid, ',')) do
      local id = tonumber(val)
      if id > 0 and not usedGpuIds[id] then
        if #Cuda.gpuIds == 0 then
          require('cutorch')
        end
        table.insert(Cuda.gpuIds, id)
        usedGpuIds[id] = true
        cutorch.setDevice(id)
        cutorch.manualSeed(opt.seed)
      else
        Cuda.cpuTraining = true
      end
      table.insert(Cuda.deviceIds, id)
    end
    if #Cuda.gpuIds > 0 then
      _G.logger:info('Using GPU(s): ' .. table.concat(Cuda.gpuIds, ', '))
      if cutorch.isCachingAllocatorEnabled and cutorch.isCachingAllocatorEnabled() then
        _G.logger:warning('The caching CUDA memory allocator is enabled. This allocator improves performance at the cost of a higher GPU memory usage. To optimize for memory, consider disabling it by setting the environment variable: THC_CACHING_ALLOCATOR=0')
      end
    end
    if Cuda.cpuTraining then
      _G.logger:info('Using CPU')
    end
    Cuda.fp16 = opt.fp16
    if Cuda.fp16 and (not cutorch or (cutorch and not cutorch.hasHalf) or Cuda.cpuTraining) then
      error("fp16 requested but installed cutorch does not support half-tensor and/or cpu training")
    end
    -- by default master node is the first one
    _G.threadDeviceId = Cuda.deviceIds[1]
  else
    _G.threadDeviceId = deviceId
    if deviceId > 0 then
      assert(deviceId <= cutorch.getDeviceCount(),
                 'GPU ' .. deviceId .. ' is requested but only '
                   .. cutorch.getDeviceCount() .. ' GPUs are available')
      cutorch.setDevice(deviceId)
    end
  end
end

-- returns RNGState for CPU and enabled GPUs
function Cuda.getRNGStates()
  local rngStates = { torch.getRNGState() }
  for _,idx in ipairs(Cuda.gpuIds) do
    table.insert(rngStates, cutorch.getRNGState(idx))
  end
  return rngStates
end

-- set RNGState from saved state
function Cuda.setRNGStates(rngStates, verbose)
  if not rngStates then
    return
  end
  if verbose then
    _G.logger:info("Restoring Random Number Generator states")
  end
  torch.setRNGState(rngStates[1])
  if #rngStates-1 ~= #Cuda.gpuIds then
    _G.logger:warning('GPU count does not match for resetting Random Number Generator - skipping')
  else
    for idx = 2, #rngStates do
      cutorch.setRNGState(rngStates[idx], idx-1)
    end
  end
end

--[[
  Recursively move all supported objects in `obj` on the GPU.
  When using CPU only, converts to float instead of the default double.
]]
function Cuda.convert(obj)
  local objtype = torch.typename(obj)
  if objtype then
    if _G.threadDeviceId > 0 and obj.cuda ~= nil then
      if objtype:find('torch%..*LongTensor') then
        return obj:type('torch.CudaLongTensor')
      elseif Cuda.fp16 then
        return obj:type('torch.CudaHalfTensor')
      else
        return obj:type('torch.CudaTensor')
      end
    elseif _G.threadDeviceId == 0 and obj.float ~= nil then
      -- Defaults to float instead of double.
      if objtype:find('torch%..*LongTensor') then
        return obj:type('torch.LongTensor')
      else
        return obj:type('torch.FloatTensor')
      end
    end
  end

  if objtype or type(obj) == 'table' then
    for k, v in pairs(obj) do
      obj[k] = Cuda.convert(v)
    end
  end

  return obj
end

--[[
  Synchronize operations on current device if working on GPU.
  Do nothing otherwise.
]]
function Cuda.synchronize()
  if _G.threadDeviceId and _G.threadDeviceId > 0 then cutorch.synchronize() end
end

--[[
  Number of available GPU.
]]
function Cuda.replicaCount()
  return #Cuda.deviceIds
end

--[[
  Free memory on the current GPU device.
]]
function Cuda.freeMemory()
  if _G.threadDeviceId > 0 then
    local freeMemory = cutorch.getMemoryUsage(_G.threadDeviceId)
    return freeMemory
  end
  return 0
end

return Cuda
