require 'torch'
require 'nn'
require 'nngraph'

local Cuda = {
  nn = nn,
  activated = false
}

function Cuda.init(opt)
  Cuda.activated = opt.gpuid > 0

  if Cuda.activated then
    local _, err = pcall(function()
      require 'cutorch'
      require 'cunn'
      if opt.cudnn then
        require 'cudnn'
        Cuda.nn = cudnn
      end
      cutorch.setDevice(opt.gpuid)
      cutorch.manualSeed(opt.seed)
    end)

    if err then
      if opt.fallback_to_cpu then
        print('Info: Failed to initialize Cuda on device ' .. opt.gpuid .. ', falling back to CPU.')
        Cuda.activated = false
      else
        error(err)
      end
    else
       print('Using GPU ' .. opt.gpuid .. '...')
    end
  end
end

function Cuda.convert(obj)
  if Cuda.activated then
    if type(obj) == 'table' then
      for i = 1, #obj do
        obj[i]:cuda()
      end
    else
      return obj:cuda()
    end
  end

  return obj
end

return Cuda
