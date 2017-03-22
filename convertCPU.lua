require('onmt.init')

local cmd = onmt.utils.ExtendedCmdLine.new('convertCPU.lua')

-- First argument define the model type: seq2seq/lm - default is seq2seq.
local modelType = cmd.getArgument(arg, '-model_type') or 'seq2seq'

local modelClass = onmt.ModelSelector(modelType)

-- Options declaration.
local options = {
  {'-data',       '', [[Path to the training *-train.t7 file from preprocess.lua]],
                      {valid=onmt.utils.ExtendedCmdLine.nonEmpty}},
  {'-save_model', '', [[Model filename (the model will be saved as
                            <save_model>_epochN_PPL.t7 where PPL is the validation perplexity]],
                      {valid=onmt.utils.ExtendedCmdLine.nonEmpty}}
}

cmd:setCmdLineOptions(options, 'Data')

onmt.Model.declareOpts(cmd)
modelClass.declareOpts(cmd)
onmt.train.Optim.declareOpts(cmd)
onmt.train.Trainer.declareOpts(cmd)
onmt.train.Checkpoint.declareOpts(cmd)

cmd:text('')
cmd:text('**Other options**')
cmd:text('')

onmt.utils.Cuda.declareOpts(cmd)
onmt.utils.Memory.declareOpts(cmd)
onmt.utils.Logger.declareOpts(cmd)
onmt.utils.Profiler.declareOpts(cmd)

cmd:option('-seed', 3435, [[Seed for random initialization]], {valid=onmt.utils.ExtendedCmdLine.isUInt()})

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

  torch.manualSeed(opt.seed)

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)
  _G.profiler = onmt.utils.Profiler.new(false)

  onmt.utils.Cuda.init(opt)
  onmt.utils.Parallel.init(opt)

  local checkpoint
  checkpoint, opt = onmt.train.Checkpoint.loadFromCheckpoint(opt)

  _G.logger:info('Converting '..modelClass.modelName()..' model')

  

  local model

  local dicts = checkpoint.dicts
  
  
  local _modelClass = onmt.ModelSelector(modelType)
  
  _G.logger:info('Load the pretrained model...') 
  local pretrained = _modelClass.load(opt, checkpoint.models, dicts)
  
  _G.logger:info('Create a new model clone (to avoid Cudnn incompatibility)...') 
  local newModel = _modelClass.new(opt, dicts, true)
  
  local pretrainedParams, _ = pretrained:initParams()
  local newParams, _ = newModel:initParams()
  
  
  _G.logger:info('Transfering the weight...') 
  for j = 1, #pretrainedParams do
    newParams[j]:copy(pretrainedParams[j])
  end
  
  checkpoint.models = {}
  
  for k, v in pairs(newModel.models) do
    if v.serialize then
      checkpoint.models[k] = v:serialize()
    else
      checkpoint.models[k] = v
    end
  end
  
  _G.logger:info('Converting model...')
  
  print(checkpoint.info)
  
  -- some of the information of the info is CudaTensor so get rid of it
  checkpoint.info = nil
  for _, model in pairs(checkpoint.models) do
		--~ print(model)
    releaseModel(model)
  end
  
  _G.logger:info('... done.')
	
	local filePath = opt.train_from .. ".cpu" 
	
  _G.logger:info('Releasing model to \'' .. filePath .. '\'...')
  torch.save(filePath, checkpoint)
  _G.logger:info('... done.')

  _G.logger:shutDown()
  
end  

main()
