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
  
  local bleuScore = _G.scorer:computeCorpusBLEU()
  
  model:training()
  _G.scorer:resetStats()
  collectgarbage()
  
  return bleuScore

end

------------------------------------------------------------------------------------------------------------------

local Trainer = torch.class('Trainer')

local options = {
  {'-save_every',              0 ,    [[Save intermediate models every this many iterations within an epoch.
                                            If = 0, will not save models within an epoch. ]],
                                      {valid=onmt.utils.ExtendedCmdLine.isUInt()}},
  {'-report_every',            100,    [[Print stats every this many iterations within an epoch.]],
                                      {valid=onmt.utils.ExtendedCmdLine.isUInt()}},
  {'-report_bleu',          false, [[Report BLEU Score for validation data.]]},
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
  _G.scorer = onmt.utils.BLEU.new(dataset.dicts.tgt.words, 4, 1)

  
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

  local function trainEpoch(epoch, doProfile)
    local epochProfiler = onmt.utils.Profiler.new(doProfile)

    local startI = self.args.start_iteration

    local numIterations = trainData:batchCount()
    -- In parallel mode, number of iterations is reduced to reflect larger batch size.
    if onmt.utils.Parallel.count > 1 and not self.args.async_parallel then
      numIterations = math.ceil(numIterations / onmt.utils.Parallel.count)
    end

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
		
		_G.profiler = onmt.utils.Profiler.new(doProfile)
		
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

    return epochState, epochProfiler:dump()
  end
  
  if self.args.report_bleu == true then
	
	local bleuScore = evalBLEU(model, validData)
	_G.logger:info('Validation BLEU score: %.2f', bleuScore)
	
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
    local validPpl = eval(model, validData)
    globalProfiler:stop('valid')

    if self.args.profiler then _G.logger:info('profile: %s', globalProfiler:log()) end
    _G.logger:info('Validation perplexity: %.2f', validPpl)

    optim:updateLearningRate(validPpl, epoch)

    checkpoint:saveEpoch(validPpl, epochState, true)
  end
end

return Trainer
