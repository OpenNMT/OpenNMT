require 'torch'
require 'nn'
require 'nngraph'

local Cuda = {
  nn = nn,
  activated = false
}

function Cuda.init(opt, gpuIdx)
  Cuda.activated = opt.gpuid > 0

  if Cuda.activated then
    local _, err = pcall(function()
      require 'cutorch'
      require 'cunn'
      if opt.cudnn then
        require 'cudnn'
        Cuda.nn = cudnn
      end
      if gpuIdx == nil then
        -- allow memory access between devices
        cutorch.getKernelPeerToPeerAccess(true)
        cutorch.manualSeedAll(opt.seed)
        cutorch.setDevice(opt.gpuid)
      else
        cutorch.setDevice(gpuIdx)
      end
      if opt.seed then
        cutorch.manualSeed(opt.seed)
      end
    end)

    if err then
      error(err)
    end
  end
end

function Cuda.convert(obj)
  if torch.typename(obj) == nil and type(obj) == 'table' then
    for i = 1, #obj do
      obj[i] = Cuda.convert(obj[i])
    end
    return obj
  end
  if Cuda.activated then
    if obj.cuda ~= nil then
        return obj:cuda()
    end
  else
    -- Defaults to float instead of double.
    return obj:float()
  end
end

function Cuda.getGPUs(ngpu)
  local gpus = {}
  if Cuda.activated then
    if ngpu > cutorch.getDeviceCount() then
      error("not enough available GPU - " .. ngpu .. " requested, " .. cutorch.getDeviceCount() .. " available")
    end
    gpus[1] = Cuda.gpuid
    local i = 1
    while #gpus ~= ngpu do
      if i ~= gpus[1] then
        table.insert(gpus, i)
      end
      i = i + 1
    end
  else
    for _ = 1, ngpu do
      table.insert(gpus, 0)
    end
  end
  return gpus
end

function Cuda.freeMemory()
  if Cuda.activated then
    local freeMemory = cutorch.getMemoryUsage(cutorch.getDevice())
    return freeMemory
  end
  return "n/a"
end

return Cuda
