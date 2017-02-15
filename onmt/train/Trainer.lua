------------------------------------------------------------------------------------------------------------------
-- Local utility functions
------------------------------------------------------------------------------------------------------------------

local function eval(model, criterion, data)
  local loss = 0
  local total = 0

  model:evaluate()

  for i = 1, data:batchCount() do
    local batch = onmt.utils.Cuda.convert(data:getBatch(i))
    loss = loss + model:forwardComputeLoss(batch, criterion)
    total = total + model:countTokens(batch)
  end

  model:training()

  return math.exp(loss / total)
end

------------------------------------------------------------------------------------------------------------------

local Trainer = torch.class("Trainer")

local trainer_options = {
  {'-save_every',              0 ,    [[Save intermediate models every this many iterations within an epoch.
                                            If = 0, will not save models within an epoch. ]],
                                      {valid=onmt.utils.ExtendedCmdLine.isUInt()}},
  {'-report_every',            50,    [[Print stats every this many iterations within an epoch.]],
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
  cmd:setCmdLineOptions(trainer_options, "Trainer")
end

function Trainer:__init(args)
  self.args = onmt.utils.ExtendedCmdLine.getModuleOpts(args, trainer_options)
  -- use profiler in Trainer
  self.args.profiler = args.profiler
  -- make a difference with options which is only used in Checkpoint
  self.options = args
end

function Trainer:train(model, optim, trainData, validData, dataset, info)
  local params, gradParams = {}, {}
  local criterion

  onmt.utils.Parallel.launch(function(idx)
    -- Only logs information of the first thread.
    local verbose = idx == 1

    -- Initialize and get model parameters.
    _G.params, _G.gradParams = _G.model:initParams(verbose)

    -- Switch to training mode.
    _G.model:training()

    -- define criterion
    _G.criterion = onmt.utils.Cuda.convert(_G.model:buildCriterion(dataset.dicts))

    -- optimize memory of the first clone
    if not self.args.disable_mem_optimization then
      local batch = onmt.utils.Cuda.convert(trainData:getBatch(1))
      batch.totalSize = batch.size
      onmt.utils.Memory.optimize(_G.model, _G.criterion, batch, verbose)
    end

    return idx, _G.criterion, _G.params, _G.gradParams
  end, function(idx, thecriterion, theparams, thegradParams)
    if idx == 1 then
      criterion = thecriterion
    end
    params[idx] = theparams
    gradParams[idx] = thegradParams
  end)

  local checkpoint = onmt.train.Checkpoint.new(self.options, model, optim, dataset.dicts)

  optim:setOptimStates(#params[1])

  local function trainEpoch(epoch, doProfile)
    local epochProfiler = onmt.utils.Profiler.new(doProfile)

    local startI = self.args.start_iteration

    local numIterations = trainData:batchCount()
    -- In parallel mode, number of iterations is reduced to reflect larger batch size.
    if onmt.utils.Parallel.count > 1 and not self.args.async_parallel then
      numIterations = math.ceil(numIterations / onmt.utils.Parallel.count)
    end

    local epochState = onmt.train.EpochState.new(epoch, numIterations, optim:getLearningRate())
    local batchOrder

    if startI > 1 and info ~= nil then
      batchOrder = info.batchOrder
    else
      -- Shuffle mini batch order.
      batchOrder = torch.randperm(trainData:batchCount())
    end

    self.args.start_iteration = 1

    if not self.args.async_parallel then
      -- synchronous parallelism or single thread
      local iter = 1
      for i = startI, trainData:batchCount(), onmt.utils.Parallel.count do
        local batches = {}
        local totalSize = 0

        for j = 1, math.min(onmt.utils.Parallel.count, trainData:batchCount() - i + 1) do
          local batchIdx = batchOrder[i + j - 1]
          if epoch <= self.args.curriculum then
            batchIdx = i + j - 1
          end
          table.insert(batches, trainData:getBatch(batchIdx))
          totalSize = totalSize + batches[#batches].size
        end

        local losses = {}

        onmt.utils.Parallel.launch(function(idx)
          _G.profiler = onmt.utils.Profiler.new(doProfile)

          _G.batch = batches[idx]
          if _G.batch == nil then
            return idx, 0, _G.profiler:dump()
          end

          -- Send batch data to the GPU.
          onmt.utils.Cuda.convert(_G.batch)
          _G.batch.totalSize = totalSize

          optim:zeroGrad(_G.gradParams)
          local loss = _G.model:trainNetwork(_G.batch, _G.criterion)

          return idx, loss, _G.profiler:dump()
        end,
        function(idx, loss, profile)
          losses[idx]=loss
          epochProfiler:add(profile)
        end)

        -- Accumulate the gradients from the different parallel threads.
        onmt.utils.Parallel.accGradParams(gradParams, batches)

        -- Update the parameters.
        optim:prepareGrad(gradParams[1], self.args.max_grad_norm)
        optim:updateParams(params[1], gradParams[1])

        -- Synchronize the parameters with the different parallel threads.
        onmt.utils.Parallel.syncParams(params)

        for bi = 1, #batches do
          epochState:update(batches[bi], losses[bi])
        end

        if iter % self.args.report_every == 0 then
          epochState:log(iter)
        end
        if self.args.save_every > 0 and iter % self.args.save_every == 0 then
          checkpoint:saveIteration(iter, epochState, batchOrder, true)
        end
        iter = iter + 1
      end
    else
      -- Asynchronous parallelism
      local counter = onmt.utils.Parallel.getCounter()
      counter:set(startI)
      local masterGPU = onmt.utils.Cuda.gpuIds[1]
      local gradBuffer = onmt.utils.Parallel.gradBuffer
      local gmutexId = onmt.utils.Parallel.gmutexId()

      while counter:get() <= trainData:batchCount() do
        local startCounter = counter:get()

        onmt.utils.Parallel.launch(function(idx)
          _G.profiler = onmt.utils.Profiler.new(doProfile)
          -- First GPU is only used for master parameters.
          -- Use 1 GPU only for 1000 first batch.
          if idx == 1 or (idx > 2 and epoch == 1 and counter:get() < self.args.async_parallel_minbatch) then
            return
          end

          local lossThread = 0
          -- Aggregate batch information.
          local batchThread = model.batchInit()
          -- Since we will be adding batches of multiple size, do as if we have a aggregated batch of size 1,
          batchThread.size = 1

          while true do
            -- Do not process more than 1000 batches (TODO - make option) in one shot.
            if counter:get() - startCounter >= 1000 then
              return lossThread, batchThread, _G.profiler:dump()
            end

            local i = counter:inc()
            if i > trainData:batchCount() then
              return lossThread, batchThread, _G.profiler:dump()
            end

            local batchIdx = batchOrder[i]
            if epoch <= self.args.curriculum then
              batchIdx = i
            end

            _G.batch = trainData:getBatch(batchIdx)

            -- Send batch data to the GPU.
            onmt.utils.Cuda.convert(_G.batch)
            _G.batch.totalSize = _G.batch.size

            optim:zeroGrad(_G.gradParams)
            local loss = model:trainNetwork(_G.batch, _G.criterion)

            -- Update the parameters.
            optim:prepareGrad(_G.gradParams)

            -- Add up gradParams to params and synchronize back to this thread.
            onmt.utils.Parallel.updateAndSync(params[1], _G.gradParams, _G.params, gradBuffer, masterGPU, gmutexId)

            batchThread = model.batchAggregate(_G.batch)
            lossThread = lossThread + loss

            -- we don't have information about the other threads here - we can only report progress
            if i % self.args.report_every == 0 then
              _G.logger:info('Epoch %d ; ... batch %d/%d', epoch, i, trainData:batchCount())
            end
          end
        end,
        function(theloss, thebatch, profile)
          if theloss then
            epochState:update(thebatch, theloss)
          end
          epochProfiler:add(profile)
        end)

        if self.args.report_every > 0 then
          epochState:log(counter:get())
        end
        if self.args.save_every > 0 then
          checkpoint:saveIteration(counter:get(), epochState, batchOrder, true)
        end
      end
    end

    return epochState, epochProfiler:dump()
  end

  _G.logger:info('Start training...')

  for epoch = self.args.start_epoch, self.args.end_epoch do
    _G.logger:info('')

    local globalProfiler = onmt.utils.Profiler.new(self.args.profiler)

    globalProfiler:start("train")
    local epochState, epochProfile = trainEpoch(epoch, self.args.profiler)
    globalProfiler:add(epochProfile)
    globalProfiler:stop("train")

    globalProfiler:start("valid")
    local validPpl = eval(model, criterion, validData)
    globalProfiler:stop("valid")

    if self.args.profiler then _G.logger:info('profile: %s', globalProfiler:log()) end
    _G.logger:info('Validation perplexity: %.2f', validPpl)

    optim:updateLearningRate(validPpl, epoch)

    checkpoint:saveEpoch(validPpl, epochState, true)
  end
end

return Trainer
