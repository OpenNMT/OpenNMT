---------------------------------------------------------------------------------
-- Local utility functions
---------------------------------------------------------------------------------

local function eval(model, data)
  local loss = 0
  local totalWords = 0

  model:evaluate()

  for i = 1, data:batchCount() do
    local batch = onmt.utils.Cuda.convert(data:getBatch(i))
    loss = loss + model:forwardComputeLoss(batch)
    totalWords = totalWords + model:getOutputLabelsCount(batch)
  end

  model:training()

  return math.exp(loss / totalWords)
end

local function evalBLEU(model, data)
  
  model:evaluate()
  
  local maxLength = onmt.Constants.MAX_TARGET_LENGTH or 50 -- to avoid nil 
  
  for i = 1, data:batchCount() do
	
	local batch = onmt.utils.Cuda.convert(data:getBatch(i))
	local sampled = model:sampleBatch(batch, maxLength, true)
	
	_G.scorer:accumulateCorpusScoreBatch(sampled, batch.targetOutput)
  end
  
  local bleuScore = _G.scorer:computeCorpusScore()
  
  model:training()
  _G.scorer:resetCorpusStats()
  collectgarbage()
  
  return bleuScore

end

------------------------------------------------------------------------------------------------------------------

local Trainer = torch.class('Trainer')

local options = {
  {'-save_every',              1000 ,    [[Save intermediate models every this many iterations within an epoch.
                                            If = 0, will not save models within an epoch. ]],
                                      {valid=onmt.utils.ExtendedCmdLine.isUInt()}},
  {'-report_every',            100,    [[Print stats every this many iterations within an epoch.]],
                                      {valid=onmt.utils.ExtendedCmdLine.isUInt()}},
  {'-loop_epoch',          true, [[Using the older training code to ensure backward-compatibility. Change this to false to use the new training code (save more frequently with bleu)]]},
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
end

function Trainer:train(model, optim, trainData, validData, dataset, info)
 _G.scorer = Rewarder(dataset.dicts.tgt.words, true, 'bleu')

  
  local verbose = true
  
  _G.model:training()
  
  _G.params, _G.gradParams = _G.model:initParams(verbose)
  
  if self.args.profiler then
    _G.model:enableProfiling()
  end
  
  -- optimize memory of the first clone
  if not self.args.disable_mem_optimization then
		local batch = onmt.utils.Cuda.convert(trainData:getBatch(1))
		batch.totalSize = batch.size
		onmt.utils.Memory.optimize(_G.model, batch, verbose)
  end
  
  

  local checkpoint = onmt.train.Checkpoint.new(self.options, model, optim, dataset.dicts)

  optim:setOptimStates(#_G.params)
  
  
  -- this function runs the model regardless of epoch
  -- and do validation for every n steps
  local function trainModel()
		local startI = self.args.start_iteration
		local numIterationsPerEpoch = trainData:batchCount()
		
		local iter = startI
		local globalIter = 1 -- tracking total steps from the beginning, regardless of current state
		
		local batchOrder
		
		if startI > 1 and info ~= nil then
      batchOrder = info.batchOrder
    else
      -- Shuffle mini batch order.
      batchOrder = torch.randperm(trainData:batchCount())
    end
		
		local currentEpoch = self.args.start_epoch
		local epochState = onmt.train.EpochState.new(currentEpoch, startI, numIterationsPerEpoch, optim:getLearningRate())
		
		assert(self.args.save_every > 0, "model must be evaluated after a number of iterations")
		
		local function validAndSave()
			_G.logger:info('')
			_G.logger:info('Doing validation ...')
			local validPpl = eval(model, validData)
			local validBleu = evalBLEU(model, validData)
			checkpoint:saveIteration(iter, numIterationsPerEpoch, epochState, batchOrder, validPpl, validBleu, true)
			optim:updateLearningRate(validPpl, currentEpoch)
			_G.logger:info('')
		end
		
		while currentEpoch <= self.args.end_epoch do
		
				
				
				local batchOrderIndex = iter 
								
				local batchIdx = batchOrder[batchOrderIndex]
				local batch = trainData:getBatch(batchIdx)
				totalSize = batch.size
				
				local losses = {}
			
				_G.batch = batch
				
				onmt.utils.Cuda.convert(_G.batch)
				
				_G.batch.totalSize = totalSize
				
				optim:zeroGrad(_G.gradParams)
				
				local loss = _G.model:trainNetwork(_G.batch)
				
				optim:prepareGrad(_G.gradParams)
				optim:updateParams(_G.params, _G.gradParams)
				
				epochState:update(model, batch, loss)
				
				if globalIter % self.args.report_every == 0 then
					epochState:log(iter)
				end
				
				if self.args.save_every > 0 and globalIter % self.args.save_every == 0 then
					validAndSave()
				end
				
				
				iter = iter + 1
				globalIter = globalIter + 1
				
				if iter > numIterationsPerEpoch then -- we start a new epoch
					iter = 1
					currentEpoch = currentEpoch + 1
					
					-- reset the stat
					epochState = onmt.train.EpochState.new(currentEpoch, 1, numIterationsPerEpoch, optim:getLearningRate())
					
					optim:updateLearningRate(math.huge, currentEpoch)
					
					-- reshuffle the training data every epoch
					batchOrder = torch.randperm(trainData:batchCount())
					
					-- save the last time
					if currentEpoch - 1 == self.args.end_epoch then
						validAndSave()
					end
					
				end
		end
  end

  local function trainEpoch(epoch, doProfile)
    --~ local epochProfiler = onmt.utils.Profiler.new(doProfile)

    local startI = self.args.start_iteration

    local numIterations = trainData:batchCount()

    local epochState = onmt.train.EpochState.new(epoch, startI, numIterations, optim:getLearningRate())
    local batchOrder

    if startI > 1 and info ~= nil then
      batchOrder = info.batchOrder
    else
      -- Shuffle mini batch order.
      batchOrder = torch.randperm(trainData:batchCount())
    end

    self.args.start_iteration = 1
    
    local iter = startI
    
    -- Looping over the batches
    for i = startI, trainData:batchCount() do
			local batches = {}
			local totalSize = 0
			
			-- Take the corresponding batch idx
			local batchIdx = batchOrder[i]
			if epoch <= self.args.curriculum then
				batchIdx = i
			end
			local batch = trainData:getBatch(batchIdx)
			totalSize = batch.size
			
			local losses = {}
			
			--~ _G.profiler = onmt.utils.Profiler.new(doProfile)
			
			_G.batch = batch
			
			onmt.utils.Cuda.convert(_G.batch)
			_G.batch.totalSize = totalSize
			
			optim:zeroGrad(_G.gradParams)
			
			local loss = _G.model:trainNetwork(_G.batch)
			
			optim:prepareGrad(_G.gradParams)
			optim:updateParams(_G.params, _G.gradParams)
			
			epochState:update(model, batch, loss)
			
			if iter % self.args.report_every == 0 then
				epochState:log(iter)
			end
					
			if self.args.save_every > 0 and iter % self.args.save_every == 0 then
				checkpoint:saveIteration(iter, epochState, batchOrder, true)
			end
			iter = iter + 1
        
    end
    

    epochState:log()
	
		return epochState
    --~ return epochState, epochProfiler:dump()
  end
  
  
	
	local bleuScore = evalBLEU(model, validData)
	_G.logger:info('Initial Validation BLEU score: %.2f', bleuScore)
	
  

  _G.logger:info('Start training...')

  for epoch = self.args.start_epoch, self.args.end_epoch do
    _G.logger:info('')
    
    if self.args.loop_epoch == true then

			local globalProfiler = onmt.utils.Profiler.new(self.args.profiler)

			globalProfiler:start('train')
			local epochState, epochProfile = trainEpoch(epoch)
			globalProfiler:add(epochProfile)
			globalProfiler:stop('train')

			globalProfiler:start('valid')
			local validPpl = eval(model, validData)
			local validBleu = evalBLEU(model, validData)
		
			
			globalProfiler:stop('valid')

			if self.args.profiler then _G.logger:info('profile: %s', globalProfiler:log()) end
			_G.logger:info('Validation perplexity: %.2f', validPpl)
			_G.logger:info('Validation BLEU score: %.2f', validBleu)

			optim:updateLearningRate(validPpl, epoch)
			--~ 
			local totalEpochs = self.args.end_epoch - self.args.start_epoch + 1
			--~ 
			--~ 
	--~ 
			checkpoint:saveEpoch(validPpl, epochState, true)
			
		else
    
			trainModel()
    
    end
  end
end

return Trainer
