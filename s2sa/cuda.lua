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
    print('Using GPU ' .. opt.gpuid .. '.')
    require 'cutorch'
    require 'cunn'
    if opt.cudnn then
      require 'cudnn'
      Cuda.nn = cudnn
    end
    cutorch.setDevice(opt.gpuid)
    cutorch.manualSeed(opt.seed)
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
