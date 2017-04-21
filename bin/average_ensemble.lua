require('onmt.init')

local cmd = onmt.utils.ExtendedCmdLine.new('average_ensemble.lua')
local pl = require('pl.import_into')()




local options = {
  {'-models', '', [[Source sequence to decode (one line per sequence)]],
               {valid=onmt.utils.ExtendedCmdLine.nonEmpty}},
  {'-output', 'pred.txt', [[Path to output the averaged model]]}
}

cmd:setCmdLineOptions(options, 'Data')


cmd:text('')
cmd:text('**Other options**')
cmd:text('')

cmd:option('-time', false, [[Measure batch translation time]])

onmt.utils.Cuda.declareOpts(cmd)
onmt.utils.Logger.declareOpts(cmd)


local function clearStateModel(model)
  for _, submodule in pairs(model.modules) do
    if torch.type(submodule) == 'table' and submodule.modules then
      clearStateModel(submodule)
    else
      submodule:clearState()
      --~ submodule:apply(function (m)
        --~ nn.utils.clear(m, 'gradWeight', 'gradBias')
      --~ end)
    end
  end
end


local function main()
  local opt = cmd:parse(arg)

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)
  onmt.utils.Cuda.init(opt)
  
  local modelFiles = pl.utils.split(opt.models, '|')
  
  
  local nModels = #modelFiles
  
  local models = {}
  
  local mainCheckpoint
  for i = 1, nModels do
		_G.logger:info('Loading \'' .. modelFiles[i] .. '\'...')
		models[i] = {}
		local checkpoint = torch.load(modelFiles[i])
		models[i].encoder = onmt.Factory.loadEncoder(checkpoint.models.encoder)
		models[i].decoder = onmt.Factory.loadDecoder(checkpoint.models.decoder)
		

		for k, v in pairs(models[i]) do
			clearStateModel(v)
		end
		 
		if i > 1 then
			checkpoint = nil
		else
			mainCheckpoint = checkpoint
			mainCheckpoint.info = nil
		end
		collectgarbage() --save memory
  end
  
  _G.logger:info('Averaring the weights of these models ...')
  
  local mainModel = models[1]
  
  for key in pairs(mainModel) do
		print(mainModel[key])
		local mainP, _ = mainModel[key]:getParameters()
		
		for i = 2, nModels do
			local subModel = models[i]
			local subP, _ = subModel[key]:getParameters()
			
			mainP:add(subP)
			
			if i == nModels then
				mainP:div(nModels)
			end
		end
  end
  
  _G.logger:info('... done.')
  
  
  
  
  local checkpoint = mainCheckpoint
  --~ checkpoint.info = nil
  
  for k, v in pairs(mainModel) do
	--~ print(v)
		--~ clearStateModel(v)
    if v.serialize then
      checkpoint.models[k] = v:serialize()
    else
      checkpoint.models[k] = v
    end
  end
  
  local filePath = opt.output
  
  _G.logger:info('Releasing model to \'' .. filePath .. '\'...')
  torch.save(filePath, checkpoint)
  _G.logger:info('... done.')

  _G.logger:shutDown()
  
  
  --~ end
  
  _G.logger:shutDown()
end

main()
