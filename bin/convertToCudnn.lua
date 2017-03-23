require('onmt.init')

local cmd = onmt.utils.ExtendedCmdLine.new('convertCPU.lua')

-- First argument define the model type: seq2seq/lm - default is seq2seq.
local modelType = cmd.getArgument(arg, '-model_type') or 'seq2seq'

local modelClass = onmt.ModelSelector(modelType)

-- Options declaration.
local options = {
  {'-save_model', '', [[Model filename (the model will be saved as
                            <save_model>_epochN_PPL.t7 where PPL is the validation perplexity]],
                      {valid=onmt.utils.ExtendedCmdLine.nonEmpty}}
}
cmd:setCmdLineOptions(options, 'Data')


onmt.Model.declareOpts(cmd)
modelClass.declareOpts(cmd)
onmt.train.Checkpoint.declareOpts(cmd)

cmd:text('')
cmd:text('**Other options**')
cmd:text('')

onmt.utils.Cuda.declareOpts(cmd)
onmt.utils.Logger.declareOpts(cmd)

cmd:option('-seed', 3435, [[Seed for random initialization]], {valid=onmt.utils.ExtendedCmdLine.isUInt()})

local opt = cmd:parse(arg)

-------------------------------------------------------------------------------
-- these functions are adapted from Michael Partheil
-- https://groups.google.com/forum/#!topic/torch7/i8sJYlgQPeA
-- the problem is that you can't call :float() on cudnn module, it won't convert
function replaceModules(net, orig_class_name, replacer)
	local nodes, container_nodes = net:findModules(orig_class_name)
		for i = 1, #nodes do
			for j = 1, #(container_nodes[i].modules) do
				if container_nodes[i].modules[j] == nodes[i] then
					local orig_mod = container_nodes[i].modules[j]
						print('replacing a cudnn module with nn equivalent...')
						print(orig_mod)
				container_nodes[i].modules[j] = replacer(orig_mod)
			end
		end
	end
end

function cudnnNetToCpu(net)
	local net_cpu = net:clone():float()
	replaceModules(net_cpu, 'cudnn.ReLU', function() return nn.ReLU() end)
	replaceModules(net_cpu, 'cudnn.Tanh', function() return nn.Tanh() end)
	replaceModules(net_cpu, 'cudnn.Sigmoid', function() return nn.Sigmoid() end)
	replaceModules(net_cpu, 'cudnn.ReLU', function() return nn.ReLU() end)
	replaceModules(net_cpu, 'cudnn.Softmax', function() return nn.Softmax() end)
	replaceModules(net_cpu, 'cudnn.LogSoftMax', function() return nn.LogSoftMax() end)
	return net_cpu
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
    model = cudnnNetToCpu(model)
  end
  
  _G.logger:info('... done.')
	
	local filePath = opt.save_model
	
	if filePath == "" then
		filePath = opt.train_from .. ".cpu" 
	end
	
  _G.logger:info('Releasing model to \'' .. filePath .. '\'...')
  torch.save(filePath, checkpoint)
  _G.logger:info('... done.')

  _G.logger:shutDown()
  
end  

main()
