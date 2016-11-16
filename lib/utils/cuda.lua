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
      if opt.seed then
        cutorch.manualSeed(opt.seed)
      end
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
  if torch.typename(obj) == 'table' then
    for i = 1, #obj do
      if Cuda.activated then
        obj[i]:cuda()
      else
        obj[i]:float()
      end
    end
  elseif Cuda.activated then
    return obj:cuda()
  else
    return obj:float()
  end
end

return Cuda
