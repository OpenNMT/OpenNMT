local Cuda = {
  gpuIds = {},
  activated = false,
  cudnn = nil
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
    local ret
    ret, Cuda.cudnn = pcall(require, 'cudnn')
    if not ret then
      _G.logger:warning("For improved efficiency with GPU - install cudnn")
      Cuda.cudnn = nil
    end

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
      cutorch.getKernelPeerToPeerAccess(true)

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
      if Cuda.cudnn and obj.modules then
        local count = 0
        local cudaobj = obj:cuda()
        -- recursively goes through the graph
        cudaobj:apply(function(m)
          if m.modules then
            for i, _ in ipairs(m.modules) do
              if torch.type(m.modules[i]) == 'nn.Sigmoid' then
                count = count + 1
                local modules=m.modules[i].modules
                -- disable recursivity in conversion since we are already recursing
                m.modules[i].modules=nil
                m.modules[i]= Cuda.cudnn.convert(m.modules[i], Cuda.cudnn)
                m.modules[i].modules=modules
              end
            end
          end
        end)
        if count > 0 then
          _G.logger:info('Using cudnn modules for ...'..torch.typename(obj)..' ('..count..')')
        end
      end
      return cudaobj
    elseif not Cuda.activated and obj.float ~= nil then
      -- Defaults to float instead of double.
      return obj:float()
    end
  end

  if torch.typename(obj) or type(obj) == 'table' then
    for k, v in pairs(obj) do
      obj[k] = Cuda.convert(v, true)
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
