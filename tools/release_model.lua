require('onmt.init')

local path = require('pl.path')

local cmd = onmt.utils.ExtendedCmdLine.new('release_model.lua')

local options = {
  {'-model', '', 'trained model file'},
  {'-output_model', '', 'released model file'},
  {'-force', false, 'force output model creation'}
}

cmd:setCmdLineOptions(options, 'Model')

cmd:text('')
cmd:text('**Other options**')
cmd:text('')

onmt.utils.Cuda.declareOpts(cmd)
onmt.utils.Logger.declareOpts(cmd)

local opt = cmd:parse(arg)

local function releaseModel(model, tensorCache)
  tensorCache = tensorCache or {}
  for _, submodule in pairs(model.modules) do
    if torch.type(submodule) == 'table' and submodule.modules then
      releaseModel(submodule, tensorCache)
    else
      submodule:float(tensorCache)
      submodule:clearState()
      submodule:apply(function (m)
        nn.utils.clear(m, 'gradWeight', 'gradBias')
        for k, v in pairs(m) do
          if type(v) == 'function' then
            m[k] = nil
          end
        end
      end)
    end
  end
end

local function main()
  assert(path.exists(opt.model), 'model \'' .. opt.model .. '\' does not exist.')

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)

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

  onmt.utils.Cuda.init(opt)

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
  for key, model in pairs(checkpoint.models) do
    if model.name == 'CudnnEncoder' then
      model = onmt.CudnnEncoder.load(model):toNN()
      checkpoint.models[key] = model
    end
    releaseModel(model)
  end
  _G.logger:info('... done.')

  _G.logger:info('Releasing model \'' .. opt.output_model .. '\'...')
  torch.save(opt.output_model, checkpoint)
  _G.logger:info('... done.')

  _G.logger:shutDown()
end

main()
