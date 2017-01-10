require('onmt.init')

local path = require('pl.path')

local cmd = torch.CmdLine()
cmd:option('-model', '', 'trained model file')
cmd:option('-output_model', '', 'released model file')
cmd:option('-gpuid', 0, [[1-based identifier of the GPU to use. CPU is used when the option is < 1]])
cmd:option('-force', false, 'force output model creation')
cmd:option('-log_file', '', [[Outputs logs to a file under this path instead of stdout.]])
cmd:option('-disable_logs', false, [[If = true, output nothing.]])
local opt = cmd:parse(arg)

local function toCPU(model)
  for _, submodule in pairs(model.modules) do
    if torch.type(submodule) == 'table' and submodule.modules then
      toCPU(submodule)
    else
      submodule:float()
      submodule:clearState()
    end
  end
end

local function main()
  assert(path.exists(opt.model), 'model \'' .. opt.model .. '\' does not exist.')

  local logFile = opt.log_file
  local mute = (opt.log_file:len() > 0)
  if opt.disable_logs then
    logFile = nil
    mute = true
  end
  _G.logger = onmt.utils.Logger.new(logFile, mute)

  if opt.output_model:len() == 0 then
    if opt.model:sub(-3) == '.t7' then
      opt.output_model = opt.model:sub(1, -4) -- copy input model without '.t7' extension
    else
      opt.output_model = opt.model
    end
    opt.output_model = opt.output_model .. '_release.t7'
  end

  if not opt.force then
    assert(not path.exists(opt.output_model),
           'output model already exists; use -force to overwrite.')
  end

  if opt.gpuid > 0 then
    require('cutorch')
    cutorch.setDevice(opt.gpuid)
  end

  _G.logger:info('Loading model \'' .. opt.model .. '\'...')

  local checkpoint
  local _, err = pcall(function ()
    checkpoint = torch.load(opt.model)
  end)
  if err then
    error('unable to load the model (' .. err .. '). If you are releasing a GPU model, it needs to be loaded on the GPU first (set -gpuid > 0)')
  end

  _G.logger:info('... done.')

  _G.logger:info('Converting model...')
  checkpoint.info = nil
  for _, model in pairs(checkpoint.models) do
    toCPU(model)
  end
  _G.logger:info('... done.')

  _G.logger:info('Releasing model \'' .. opt.output_model .. '\'...')
  torch.save(opt.output_model, checkpoint)
  _G.logger:info('... done.')

  _G.logger:shutDown()
end

main()
