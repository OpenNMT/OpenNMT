---------------------------------------------------------------------------------
-- Local utility functions
---------------------------------------------------------------------------------

local function eval(model, data)
  local loss = 0
  local totalWords = 0
  local totalSize = 0
  local totalAccurate = 0
  local nSentence = 0
  
  local lenError = 0
  model:evaluate()

  for i = 1, data:batchCount() do
    local batch = onmt.utils.Cuda.convert(data:getBatch(i))
    local pLoss, nAccuracy, expectedLength = model:evalLenPredictor(batch)
    --~ loss = loss + model:evalLenPredictor(batch)
    loss = loss + pLoss
    totalAccurate = totalAccurate + nAccuracy
    --~ totalWords = totalWords + model:getOutputLa	belsCount(batch)
    totalSize =  totalSize + batch.size
    
    --~ print(expectedLength)
    
    lenError = lenError + torch.abs( expectedLength - batch.targetSize ):sum(1):squeeze()
    
    nSentence = nSentence + batch.size
  end

  model:training()
  totalAccurate = totalAccurate / totalSize
  
  lenError = lenError / nSentence

  return math.exp(loss / totalSize), totalAccurate, lenError
end

------------------------------------------------------------------------------------------------------------------

local Trainer = torch.class('LengthTrainer')

local options = {
  {'-save_every',              0 ,    [[Save intermediate models every this many iterations within an epoch.
                                            If = 0, will not save models within an epoch. ]],
                                      {valid=onmt.utils.ExtendedCmdLine.isUInt()}},
  {'-report_every',            100,    [[Print stats every this many iterations within an epoch.]],
                                      {valid=onmt.utils.ExtendedCmdLine.isUInt()}},
  {'-async_parallel',          false, [[Use asynchronous parallelism training.]]},
  {'-async_parallel_minbatch', 1000,  [[For async parallel computing, minimal number of batches before being parallel.]],
                                      {valid=onmt.utils.ExtendedCmdLine.isUInt()}},
  {'-start_iteration',         1,     [[If loading from a checkpoint, the iteration from which to start]],
                                         {valid=onmt.utils.ExtendedCmdLine.isInt(1)}},
  {'-end_epoch',               13,    [[The final epoch of the training]],
                                      {valid=onmt.utils.ExtendedCmdLine.isInt(1)}},
  {'-start_epoch',             1,     [[If loading from a checkpoint, the epoch from which to start]],
                                      {valid=onmt.utils.ExtendedCmdLine.isInt(1)}},
  {'-curriculum',              0,     [[For this many epochs, order the minibatches based on source
                                            sequence length. Sometimes setting this to 1 will increase convergence speed.]],
                                      {valid=onmt.utils.ExtendedCmdLine.isUInt()}}
}

function Trainer.declareOpts(cmd)
  cmd:setCmdLineOptions(options, 'Trainer')
end

function Trainer:__init(args)
  self.args = onmt.utils.ExtendedCmdLine.getModuleOpts(args, options)
  -- Use profiler in Trainer.
  self.args.profiler = args.profiler
  -- Make a difference with options which is only used in Checkpoint.
  self.options = args
  
  self.bestValid = 100000
end

function Trainer:train(model, optim, trainData, validData, dataset, info)
	
	local params, gradParams = {}, {}
	
	local verbose = true
	_G.params, _G.gradParams = _G.model:initParams(verbose)
	_G.model:training()
	
	if self.args.profiler then
		_G.model:enableProfiling()
	end
	
	if not self.args.disable_mem_optimization then
		local batch = onmt.utils.Cuda.convert(trainData:getBatch(1))
		batch.totalSize = batch.size
		onmt.utils.Memory.optimize(_G.model, batch, verbose)
	end
	
	local checkpoint = onmt.train.Checkpoint.new(self.options, model, optim, dataset.dicts)
	checkpoint:setMode('best') -- only save best epoches
	
	optim:setOptimStates(#_G.params)
	
	local function trainEpoch(epoch, doProfile)
		
		local epochProfiler = onmt.utils.Profiler.new(doProfile)
		local startI = self.args.start_iteration
		local numIterations = trainData:batchCount()
		
		local epochState = onmt.train.EpochState.new(epoch, startI, numIterations, optim:getLearningRate())
		epochState:setMode('length')
		local batchOrder
		
		if startI > 1 and info ~= nil then
			batchOrder = info.batchOrder
		else
		  -- Shuffle mini batch order.
		  batchOrder = torch.randperm(trainData:batchCount())
		end
		
		-- Start training
		local iter = startI
		for i = startI, trainData:batchCount() do
			local totalSize = 0
			
			local batchIdx = batchOrder[i]
			if epoch <= self.args.curriculum then
				batchIdx = i
			end
						
			_G.batch = trainData:getBatch(batchIdx)
			totalSize = totalSize + _G.batch.size
			
			local losses = {}
			
			onmt.utils.Cuda.convert(_G.batch)
			_G.batch.totalSize = totalSize
			_G.profiler = onmt.utils.Profiler.new(doProfile)
			
			optim:zeroGrad(_G.gradParams)
			
			--~ local loss = _G.model:trainNetwork(_G.batch)
			local loss = _G.model:trainLenPredictor(_G.batch)
			
			optim:prepareGrad(_G.gradParams)
			optim:updateParams(_G.params, _G.gradParams)
			
			epochState:update(model, _G.batch, loss)
			
			if iter % self.args.report_every == 0 then
			  epochState:log(iter)
			end
			
			if self.args.save_every > 0 and iter % self.args.save_every == 0 then
			  checkpoint:saveIteration(iter, epochState, batchOrder, true)
			end
			
			iter = iter + 1
			
		end
		
		return epochState, epochProfiler:dump()
	end
	
	_G.logger:info('Start training...')
	
	for epoch = self.args.start_epoch, self.args.end_epoch do
		_G.logger:info('')

		local globalProfiler = onmt.utils.Profiler.new(self.args.profiler)

		globalProfiler:start('train')
		local epochState, epochProfile = trainEpoch(epoch, self.args.profiler)
		globalProfiler:add(epochProfile)
		globalProfiler:stop('train')

		globalProfiler:start('valid')
		local validPpl, validAccuracy, lenError = eval(model, validData)
		--~ assert(lenError:size(1) == 1)
		
		
		globalProfiler:stop('valid')

		if self.args.profiler then _G.logger:info('profile: %s', globalProfiler:log()) end
		_G.logger:info('Validation perplexity: %.2f', validPpl)
		_G.logger:info('Validation Accuracy: %.2f', validAccuracy)
		_G.logger:info('Expected length error: %.2f', lenError)
		--~ print(lenError)
		
		local mainMetric = lenError
		optim:updateLearningRate(mainMetric, epoch)
		
		if self.bestValid > mainMetric then
			self.bestValid = lenError
			checkpoint:saveEpoch(self.bestValid, epochState, true)
		end
		
		
	end
	
	
end

return Trainer
