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
      print('cudnn activated')
      require 'cudnn'
      Cuda.nn = cudnn
    else
      print('no cuddn')
    end
    cutorch.setDevice(opt.gpuid)
    cutorch.manualSeed(opt.seed)
  end
end

function Cuda.convert(networks)
  if Cuda.activated then
    for _, net in networks do
      net:cuda()
    end
  end
end

return Cuda
