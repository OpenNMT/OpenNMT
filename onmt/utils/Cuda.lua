local Cuda = {
  gpuIds = {},
  activated = false,
  cudnn = nil,
  _cudnnModule = {}
}

local cuda_options = {
  {'-gpuid',     0,   [[List of comma-separated GPU identifiers (1-indexed). CPU is used when set to 0.]],
                                 {valid=onmt.ExtendedCmdLine.isUInt()}},
  {'-no_nccl', false, [[Disable usage of nccl in parallel mode.]]},
  {'-cudnn', '', [[Layers, comma-separated, for which you want to use optimized cudnn routines]],
                                 {enum={'', 'RNN', 'SoftMax', 'Sigmoid'}}}
}

function Cuda.declareOpts(cmd)
  cmd:setCmdLineOptions(cuda_options)
end

function Cuda.init(opt, masterGPU)
  for _, val in ipairs(onmt.utils.String.split(tostring(opt.gpuid), ',')) do
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

    if opt.cudnn ~= '' then
      -- first check cudnn is available
      ret, Cuda.cudnn = pcall(require, 'cudnn')
      if not ret then
        _G.logger:warning("-cudnn option only works with cudnn library - library not found: disabling the option")
        Cuda.cudnn = nil
      else
        local modules = onmt.utils.String.split(opt.cudnn, ",")
        for _,k in ipairs(modules) do
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

      cutorch.manualSeedAll(opt.seed)
    end

    cutorch.setDevice(Cuda.gpuIds[masterGPU])

    if opt.seed then
      cutorch.manualSeed(opt.seed)
    end
  end
end

local function _cudnnSupportedNN(name)
  -- do not use cudnn SoftMax for attention which is too small for getting any gain
  return (name == 'nn.LogSoftMax' and Cuda.cudnnSupport('SoftMax'))
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
                m.modules[i].algorithm = 'CUDNN_SOFTMAX_FAST'
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

function Cuda.cudnnSupport(module)
  return (Cuda.cudnn and Cuda._cudnnModule[module]) or nil
end

return Cuda
