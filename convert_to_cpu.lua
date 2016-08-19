require 'nn'
require 'string'
require 'hdf5'
require 'nngraph'
require 'cunn'
require 'cutorch'

require 'models'
require 'data'
require 'util'

cmd = torch.CmdLine()

-- file location
cmd:option('-gpu_file', 'gpu_model.t7','gpu model file')
cmd:option('-cpu_file', 'cpu_model.t7', 'cpu output file')
cmd:option('-gpuid', 2, 'which gpuid to use')
opt = cmd:parse(arg)

function main()
  print('loading gpu model ' .. opt.gpu_file)
  checkpoint = torch.load(opt.gpu_file)
  model, model_opt = checkpoint[1], checkpoint[2]
  if model_opt.cudnn == 1 then
    require 'cudnn'
  end
  cutorch.setDevice(opt.gpuid)
  for i = 1, #model do
    model[i]:double()
  end
  print('saving cpu model to ' .. opt.cpu_file)
  torch.save(opt.cpu_file, {model, model_opt})
end
main()

