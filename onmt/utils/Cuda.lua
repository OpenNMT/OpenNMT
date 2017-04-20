local ExtendedCmdLine = require('onmt.utils.ExtendedCmdLine')

local Cuda = {
  fp16 = false,
  gpuIds = {},
  activated = false
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

function Cuda.init(opt, masterGPU)
  for _, val in ipairs(onmt.utils.String.split(opt.gpuid, ',')) do
    local id = tonumber(val)
    assert(id ~= nil and id >= 0, 'invalid GPU identifier: ' .. val)
    if id > 0 then
      table.insert(Cuda.gpuIds, id)
    end
  end

  Cuda.activated = #Cuda.gpuIds > 0

  if Cuda.activated then
    local _, err = pcall(function()
      require('cutorch')
      require('cunn')
      Cuda.fp16 = opt.fp16

      if masterGPU == nil then
        masterGPU = 1

        -- Validate GPU identifiers.
        for i = 1, #Cuda.gpuIds do
          assert(Cuda.gpuIds[i] <= cutorch.getDeviceCount(),
                 'GPU ' .. Cuda.gpuIds[i] .. ' is requested but only '
                   .. cutorch.getDeviceCount() .. ' GPUs are available')
        end

        _G.logger:info('Using GPU(s): ' .. table.concat(Cuda.gpuIds, ', '))

        if cutorch.isCachingAllocatorEnabled and cutorch.isCachingAllocatorEnabled() then
          _G.logger:warning('The caching CUDA memory allocator is enabled. This allocator improves performance at the cost of a higher GPU memory usage. To optimize for memory, consider disabling it by setting the environment variable: THC_CACHING_ALLOCATOR=0')
        end

      end

      cutorch.setDevice(Cuda.gpuIds[masterGPU])

      if opt.seed then
        cutorch.manualSeed(opt.seed)
      end
    end)

    if err then
      if opt.fallback_to_cpu then
        _G.logger:warning('Falling back to CPU')
        Cuda.activated = false
      else
        error(err)
      end
    end
    if Cuda.fp16 and not cutorch.hasHalf then
      error("installed cutorch does not support half-tensor")
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
function Cuda.setRNGStates(rngStates)
  if not rngStates then
    return
  end
  _G.logger:info("Restoring random number generator states...")
  torch.setRNGState(rngStates[1])
  if #rngStates-1 ~= #Cuda.gpuIds then
    _G.logger:warning('GPU count does not match for resetting random number generator. Skipping.')
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
    if Cuda.activated and obj.cuda ~= nil then
      if objtype:find('torch%..*LongTensor') then
        return obj:type('torch.CudaLongTensor')
      elseif Cuda.fp16 then
        return obj:type('torch.CudaHalfTensor')
      else
        return obj:type('torch.CudaTensor')
      end
    elseif not Cuda.activated and obj.float ~= nil then
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
  if Cuda.activated then cutorch.synchronize() end
end

--[[
  Number of available GPU.
]]
function Cuda.gpuCount()
  return #Cuda.gpuIds
end

--[[
  Free memory on the current GPU device.
]]
function Cuda.freeMemory()
  if Cuda.activated then
    local freeMemory = cutorch.getMemoryUsage(cutorch.getDevice())
    return freeMemory
  end
  return 0
end

return Cuda
