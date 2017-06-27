require('onmt.init')

local path = require('pl.path')

local cmd = onmt.utils.ExtendedCmdLine.new('average_models.lua')

local options = {
  {
    '-models', {} ,
    [[Path to models to average.]]
  },
  {
    '-output_model', 'final_averaged.t7',
    [[Path to the averaged model.]]
  },
  {
    '-force', false,
    [[Force output model creation even if the target file exists.]]
  }
}

cmd:setCmdLineOptions(options, 'Model')

onmt.utils.Cuda.declareOpts(cmd)
onmt.utils.Logger.declareOpts(cmd)

local opt = cmd:parse(arg)

local function appendParameters(store, params)
  for _, p in ipairs(params) do
    table.insert(store, p)
  end
end

local function gatherModelParameters(model, store)
  store = store or {}

  for _, submodule in pairs(model.modules) do
    if torch.type(submodule) == 'table' and submodule.modules then
      gatherModelParameters(submodule, store)
    else
      appendParameters(store, submodule:parameters())
    end
  end

  return store
end

local function gatherParameters(models)
  local parameters = {}

  for _, model in pairs(models) do
    appendParameters(parameters, gatherModelParameters(model))
  end

  return parameters
end


local function main()

  for _, f in ipairs(opt.models) do
    assert(path.exists(f), 'model \'' .. f .. '\' does not exist.')
  end

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)

  if not opt.force then
    assert(not path.exists(opt.output_model),
           'output model already exists; use -force to overwrite.')
  end

  onmt.utils.Cuda.init(opt)

  local checkpoint1
  local averageParams

  for k, f in ipairs(opt.models) do
    _G.logger:info('Loading model \'' .. f .. '\'...')
    if k == 1 then
      local _, err = pcall(function ()
        checkpoint1 = torch.load(f)
      end)
      if err then
        error('unable to load the model (' .. err .. ').')
      end
      averageParams = gatherParameters(checkpoint1.models)
    else
      local checkpoint
      local _, err = pcall(function ()
        checkpoint = torch.load(f)
      end)
      if err then
        error('unable to load the model (' .. err .. ').')
      end
      local params = gatherParameters(checkpoint.models)
      for i = 1, #params do
        averageParams[i]:mul(k-1):add(params[i]):div(k)
      end
    end
    _G.logger:info('... done.')
  end

  _G.logger:info('Saving model \'' .. opt.output_model .. '\'...')
  torch.save(opt.output_model, checkpoint1)
  _G.logger:info('... done.')

  _G.logger:shutDown()
end

main()
