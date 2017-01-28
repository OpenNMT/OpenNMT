local function eval(model, criterion, data)
  local loss = 0
  local total = 0

  model:evaluate()

  for i = 1, data:batchCount() do
    local batch = onmt.utils.Cuda.convert(data:getBatch(i))
    loss = loss + model:forwardComputeLoss(batch, criterion)
    total = total + batch.targetNonZeros
  end

  model:training()

  return math.exp(loss / total)
end

local function train(opt, model, optim, trainData, validData, dataset, info)
  local params, gradParams = {}, {}
  local criterion

  onmt.utils.Parallel.launch(function(idx)
    -- Only logs information of the first thread.
    local verbose = idx == 1 and not opt.json_log

    -- Initialize and get model parameters.
    _G.params, _G.gradParams = _G.model:initParams(opt, verbose)

    -- Switch to training mode.
    _G.model:training()

    -- define criterion
    _G.criterion = onmt.utils.Cuda.convert(_G.model:buildCriterion(dataset))

    -- optimize memory of the first clone
    if not opt.disable_mem_optimization then
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

  local checkpoint = onmt.train.Checkpoint.new(opt, model, optim, dataset.dicts)

  optim:setOptimStates(#params[1])

  local function trainEpoch(epoch, lastValidPpl, doProfile)
    local epochState
    local batchOrder

    local epochProfiler = onmt.utils.Profiler.new(doProfile)

    local startI = opt.start_iteration

    local numIterations = trainData:batchCount()
    -- In parallel mode, number of iterations is reduced to reflect larger batch size.
    if onmt.utils.Parallel.count > 1 and not opt.async_parallel then
      numIterations = math.ceil(numIterations / onmt.utils.Parallel.count)
    end

    if startI > 1 and info ~= nil then
      epochState = onmt.train.EpochState.new(epoch, numIterations, optim:getLearningRate(), lastValidPpl, info.epochStatus)
      batchOrder = info.batchOrder
    else
      epochState = onmt.train.EpochState.new(epoch, numIterations, optim:getLearningRate(), lastValidPpl)
      -- Shuffle mini batch order.
      batchOrder = torch.randperm(trainData:batchCount())
    end

    opt.start_iteration = 1

    if not opt.async_parallel then
      local iter = 1
      for i = startI, trainData:batchCount(), onmt.utils.Parallel.count do
        local batches = {}
        local totalSize = 0

        for j = 1, math.min(onmt.utils.Parallel.count, trainData:batchCount() - i + 1) do
          local batchIdx = batchOrder[i + j - 1]
          if epoch <= opt.curriculum then
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

          optim:zeroGrad(_G._gradParams)
          local loss = model:trainNetwork(_G.batch, _G.criterion, doProfile)

          return idx, loss, _G.profiler:dump()
        end,
        function(idx, loss, profile)
          losses[idx]=loss
          epochProfiler:add(profile)
        end)

        -- Accumulate the gradients from the different parallel threads.
        onmt.utils.Parallel.accGradParams(gradParams, batches)

        -- Update the parameters.
        optim:prepareGrad(gradParams[1], opt.max_grad_norm)
        optim:updateParams(params[1], gradParams[1])

        -- Synchronize the parameters with the different parallel threads.
        onmt.utils.Parallel.syncParams(params)

        for bi = 1, #batches do
          epochState:update(batches[bi], losses[bi])
        end

        if iter % opt.report_every == 0 then
          epochState:log(iter, opt.json_log)
        end
        if opt.save_every > 0 and iter % opt.save_every == 0 then
          checkpoint:saveIteration(iter, epochState, batchOrder, not opt.json_log)
        end
        iter = iter + 1
      end
    else
      -- Synchronous parallelism.
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
          if idx == 1 or (idx > 2 and epoch == 1 and counter:get() < opt.async_parallel_minbatch) then
            return
          end

          local lossThread = 0
          local batchThread = {
            size = 1,
            sourceLength = 0,
            targetLength = 0,
            targetNonZeros = 0
          }

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
            if epoch <= opt.curriculum then
              batchIdx = i
            end

            _G.batch = trainData:getBatch(batchIdx)

            -- Send batch data to the GPU.
            onmt.utils.Cuda.convert(_G.batch)
            _G.batch.totalSize = _G.batch.size

            optim:zeroGrad(_G._gradParams)
            local loss = model:trainNetwork(_G.batch, _G.criterion, doProfile)

            -- Update the parameters.
            optim:prepareGrad(_G.gradParams, opt.max_grad_norm)

            -- Add up gradParams to params and synchronize back to this thread.
            onmt.utils.Parallel.updateAndSync(params[1], _G.gradParams, _G.params, gradBuffer, masterGPU, gmutexId)

            batchThread.sourceLength = batchThread.sourceLength + _G.batch.sourceLength * _G.batch.size
            batchThread.targetLength = batchThread.targetLength + _G.batch.targetLength * _G.batch.size
            batchThread.targetNonZeros = batchThread.targetNonZeros + _G.batch.targetNonZeros
            lossThread = lossThread + loss

            -- we don't have information about the other threads here - we can only report progress
            if i % opt.report_every == 0 then
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

        if opt.report_every > 0 then
          epochState:log(counter:get(), opt.json_log)
        end
        if opt.save_every > 0 then
          checkpoint:saveIteration(counter:get(), epochState, batchOrder, not opt.json_log)
        end
      end
    end

    return epochState, epochProfiler:dump()
  end

  local validPpl = 0

  if not opt.json_log then
    _G.logger:info('Start training...')
  end

  for epoch = opt.start_epoch, opt.end_epoch do
    if not opt.json_log then
      _G.logger:info('')
    end

    local globalProfiler = onmt.utils.Profiler.new(opt.profiler)

    globalProfiler:start("train")
    local epochState, epochProfile = trainEpoch(epoch, validPpl, opt.profiler)
    globalProfiler:add(epochProfile)
    globalProfiler:stop("train")

    globalProfiler:start("valid")
    validPpl = eval(model, criterion, validData)
    globalProfiler:stop("valid")

    if not opt.json_log then
      if opt.profiler then _G.logger:info('profile: %s', globalProfiler:log()) end
      _G.logger:info('Validation perplexity: %.2f', validPpl)
    end

    optim:updateLearningRate(validPpl, epoch)

    checkpoint:saveEpoch(validPpl, epochState, not opt.json_log)
  end
end

return { train = train }
