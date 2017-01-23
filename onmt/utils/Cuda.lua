local Cuda = {
  gpuIds = {},
  activated = false
}

function Cuda.declareOpts(cmd)
  cmd:option('-gpuid', '0', [[List of comma-separated GPU identifiers (1-indexed). CPU is used when set to 0.]])
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
    require('cutorch')
    require('cunn')

    if masterGPU == nil then
      masterGPU = 1

      -- Validate GPU identifiers.
      for i = 1, #Cuda.gpuIds do
        assert(Cuda.gpuIds[i] <= cutorch.getDeviceCount(),
               'GPU ' .. Cuda.gpuIds[i] .. ' is requested but only '
                 .. cutorch.getDeviceCount() .. ' GPUs are available')
      end

      _G.logger:info('Using GPU(s): ' .. table.concat(Cuda.gpuIds, ', '))

      -- Allow memory access between devices.
      cutorch.setKernelPeerToPeerAccess(true)

      if opt.seed then
        cutorch.manualSeedAll(opt.seed)
      end
    end

    cutorch.setDevice(Cuda.gpuIds[masterGPU])

    if opt.seed then
      cutorch.manualSeed(opt.seed)
    end
  end
end

--[[
  Recursively move all supported objects in `obj` on the GPU.
  When using CPU only, converts to float instead of the default double.
]]
function Cuda.convert(obj)
  if torch.typename(obj) then
    if Cuda.activated and obj.cuda ~= nil then
      return obj:cuda()
    elseif not Cuda.activated and obj.float ~= nil then
      -- Defaults to float instead of double.
      return obj:float()
    end
  end

  if torch.typename(obj) or type(obj) == 'table' then
    for k, v in pairs(obj) do
      obj[k] = Cuda.convert(v)
    end
  end

  return obj
end

function Cuda.gpuCount()
  return #Cuda.gpuIds
end

function Cuda.freeMemory()
  if Cuda.activated then
    local freeMemory = cutorch.getMemoryUsage(cutorch.getDevice())
    return freeMemory
  end
  return 0
end

return Cuda
