local Cuda = {
  gpuIds = {},
  activated = false,
  cudnn = nil,
  _cudnnModule = nil
}

function Cuda.declareOpts(cmd)
  cmd:option('-gpuid', '0', [[List of comma-separated GPU identifiers (1-indexed). CPU is used when set to 0.]])
  cmd:option('-cudnn', nil, [[Layers, comma-separated, for which you want to use optimized cudnn routines: RNN, SoftMax, Activation.]])
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

    if opt.cudnn then
      -- first check cudnn is available
      ret, Cuda.cudnn = pcall(require, 'cudnn')
      if not ret then
        _G.logger:warning("-cudnn option only works with cudnn library - library not found: disabling the option")
        Cuda.cudnn = nil
      else
        local modules = onmt.utils.String.split(opt.cudnn, ",")
        for k in modules do
          Cuda._cudnnModule[k] = true
        end
      end
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

local function _cudnnSupportedNN(name)
  return ((name == 'nn.SoftMax' or name == 'nn.LogSoftMax') and Cuda.cudnnSupport('Softmax'))
         or ((name == 'nn.Sigmoid' or name == 'nn.Tanh' or name == 'nn.ReLU') and Cuda.cudnnSupport('Activation'))
end

--[[
  Recursively move all supported objects in `obj` on the GPU.
  When using CPU only, converts to float instead of the default double.
]]
function Cuda.convert(obj)
  if torch.typename(obj) then
    if Cuda.activated and obj.cuda ~= nil then
      local cudaobj = obj:cuda()
      if Cuda.cudnn and obj.modules then
        local count = 0
        -- recursively goes through the graph
        cudaobj:apply(function(m)
          if m.modules then
            for i, _ in ipairs(m.modules) do
              if _cudnnSupportedNN(torch.type(m.modules[i])) then
                count = count + 1
                local modules = m.modules[i].modules
                -- disable recursivity in conversion since we are already recursing
                m.modules[i].modules = nil
                m.modules[i] = Cuda.cudnn.convert(m.modules[i], Cuda.cudnn)
                m.modules[i].modules = modules
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

function Cuda.cudnnSupport(module)
  return (Cuda.cudnn and Cuda._cudnnModule[module]) or nil
end

return Cuda
