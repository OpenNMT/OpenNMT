local Trainer = torch.class('Trainer')

local options = {
  {
    '-save_every', 5000,
    [[Save intermediate models every this many iterations within an epoch.
      If = 0, will not save intermediate models.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt()
    }
  },
  {
    '-report_every', 50,
    [[Report progress every this many iterations within an epoch.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt()
    }
  },
  {
    '-async_parallel', false,
    [[When training on multiple GPUs, update parameters asynchronously.]]
  },
  {
    '-async_parallel_minbatch', 1000,
    [[In asynchronous training, minimal number of sequential batches before being parallel.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt()
    }
  },
  {
    '-start_iteration', 1,
    [[If loading from a checkpoint, the iteration from which to start.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isInt(1)
    }
  },
  {
    '-start_epoch', 1,
    [[If loading from a checkpoint, the epoch from which to start.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isInt(1)
    }
  },
  {
    '-end_epoch', 13,
    [[The final epoch of the training.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isInt(1)
    }
  },
  {
    '-curriculum', 0,
    [[For this many epochs, order the minibatches based on source length (from smaller to longer).
      Sometimes setting this to 1 will increase convergence speed.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt(),
      train_state = true
    }
  }
}

function Trainer.declareOpts(cmd)
  cmd:setCmdLineOptions(options, 'Trainer')
  onmt.train.Optim.declareOpts(cmd)
end

function Trainer:__init(args, model, dicts, firstBatch)
  self.args = onmt.utils.ExtendedCmdLine.getModuleOpts(args, options)
  self.args.profiler = args.profiler
  self.args.disable_mem_optimization = args.disable_mem_optimization

  self.optim = onmt.train.Optim.new(args)
  self.saver = onmt.train.Saver.new(args, model, self.optim, dicts)

  self.model = model

  -- Set training mode.
  model:training()

  local params
  local gradParams

  if not onmt.train.Saver.checkpointDefined(args) then
    params, gradParams = model:initParams()

    -- Add profiling hooks.
    if self.args.profiler then
      model:enableProfiling()
    end

    -- If enabled, share internal buffers to optimize for memory.
    if not self.args.disable_mem_optimization then
      if not firstBatch then
        _G.logger:error('A first batch is needed to optimize the computation graph for memory')
      else
        onmt.utils.Memory.optimize(model, onmt.utils.Cuda.convert(firstBatch), verbose)
      end
    end
  else
    params, gradParams = model:getParams()
  end

  self.params = {}
  self.gradParams = {}

  -- Create network replicas.
  onmt.utils.Parallel.launch(function(idx)
    if idx == 1 then
      -- First replica is the reference model.
      _G.model = model
      _G.params, _G.gradParams = params, gradParams
    else
      _G.model = onmt.utils.Tensor.deepClone(model)
      _G.params, _G.gradParams = params:clone(), gradParams:clone()
    end
    return idx, _G.params, _G.gradParams
  end, function(idx, params, gradParams)
    self.params[idx] = params
    self.gradParams[idx] = gradParams
  end)
end

function Trainer:eval(data)
  local loss = 0
  local totalWords = 0

  self.model:evaluate()

  for i = 1, data:batchCount() do
    local batch = onmt.utils.Cuda.convert(data:getBatch(i))
    loss = loss + self.model:forwardComputeLoss(batch)
    totalWords = totalWords + self.model:getOutputLabelsCount(batch)
  end

  self.model:training()

  return math.exp(loss / totalWords)
end

function Trainer:trainEpoch(data, epoch, startIteration, batchOrder)
  local function getBatchIdx(idx)
    return batchOrder and batchOrder[idx] or idx
  end

  startIteration = startIteration or 1

  if data.sample then
    data:sample()
  end

  local numIterations = data:batchCount()
  -- In parallel mode, the number of iterations is reduced to reflect larger batch size.
  if onmt.utils.Parallel.count > 1 and not self.args.async_parallel then
    numIterations = math.ceil(numIterations / onmt.utils.Parallel.count)
  end

  local epochState = onmt.train.EpochState.new(epoch, startIteration, numIterations, self.optim:getLearningRate())
  local epochProfiler = onmt.utils.Profiler.new(self.args.profiler)
  epochProfiler:add('train')

  local needLog = false

  if not self.args.async_parallel then
    -- Synchronous training.
    local iter = startIteration
    for i = startIteration, data:batchCount(), onmt.utils.Parallel.count do
      local batches = {}
      local totalSize = 0
      needLog = true

      for j = 1, math.min(onmt.utils.Parallel.count, data:batchCount() - i + 1) do
        local batchIdx = getBatchIdx(i + j - 1)
        table.insert(batches, data:getBatch(batchIdx))
        totalSize = totalSize + batches[#batches].size
      end

      local losses = {}
      local indvAvgLosses = {}

      onmt.utils.Parallel.launch(function(idx)
        _G.profiler = onmt.utils.Profiler.new(self.args.profiler)

        _G.batch = batches[idx]
        if _G.batch == nil then
          return idx, 0, nil, _G.profiler:dump()
        end

        -- Send batch data to the GPU.
        onmt.utils.Cuda.convert(_G.batch)
        _G.batch.totalSize = totalSize

        self.optim:zeroGrad(_G.gradParams)
        local loss, indvAvgLoss = _G.model:trainNetwork(_G.batch)

        return idx, loss, indvAvgLoss, _G.profiler:dump()
      end,
      function(idx, loss, indvAvgLoss, profile)
        losses[idx] = loss
        if data.needIndividualLosses and data:needIndividualLosses() then
          indvAvgLosses[idx] = indvAvgLoss
        end
        epochProfiler:add(profile)
      end)

      -- Accumulate the gradients from the different parallel threads.
      onmt.utils.Parallel.accGradParams(self.gradParams, batches)

      -- Update the parameters.
      self.optim:prepareGrad(self.gradParams[1])
      self.optim:updateParams(self.params[1], self.gradParams[1])

      -- Synchronize the parameters with the different parallel threads.
      onmt.utils.Parallel.syncParams(self.params)

      for bi = 1, #batches do
        epochState:update(self.model, batches[bi], losses[bi])
        if data.needIndividualLosses and data:needIndividualLosses() then
          data:setLoss(getBatchIdx(i + bi - 1), indvAvgLosses[bi])
        end
      end

      if iter % self.args.report_every == 0 then
        epochState:log(iter)
        needLog = false
      end
      if self.args.save_every > 0 and iter % self.args.save_every == 0 then
        self.saver:saveIteration(iter, epochState, batchOrder, true)
      end
      iter = iter + 1
    end
  else
    -- Asynchronous training.
    local counter = onmt.utils.Parallel.getCounter()
    local masterGPU = onmt.utils.Cuda.gpuIds[1]
    local gradBuffer = onmt.utils.Parallel.gradBuffer
    local gmutexId = onmt.utils.Parallel.gmutexId()

    local maxConcurrentIter = self.args.report_every
    if self.args.save_every > 0 and self.args.save_every < maxConcurrentIter then
      maxConcurrentIter = self.args.save_every
    end
    local iter = 0

    counter:set(startIteration)

    while counter:get() <= data:batchCount() do
      needLog = true
      local startCounter = counter:get()

      onmt.utils.Parallel.launch(function(idx)
        _G.profiler = onmt.utils.Profiler.new(self.args.profiler)
        -- First GPU is only used for master parameters.
        -- Use 1 GPU only for 1000 first batch.
        if idx == 1 or (idx > 2 and epoch == 1 and counter:get() < self.args.async_parallel_minbatch) then
          return
        end

        local batches = {}
        local losses = {}
        local indvAvgLosses = {}

        while true do
          local i = counter:inc()
          if i - startCounter >= maxConcurrentIter or i > data:batchCount() then
            return batches, losses, indvAvgLosses, _G.profiler:dump()
          end

          local batchIdx = getBatchIdx(i)

          _G.batch = data:getBatch(batchIdx)
          table.insert(batches, onmt.utils.Tensor.deepClone(_G.batch))
          onmt.utils.Cuda.convert(_G.batch)

          self.optim:zeroGrad(_G.gradParams)
          local loss, indvAvgLoss = _G.model:trainNetwork(_G.batch)
          table.insert(losses, loss)
          if data.needIndividualLosses and data:needIndividualLosses() then
            indvAvgLosses[batchIdx] = indvAvgLoss
          end

            -- Update the parameters.
          self.optim:prepareGrad(_G.gradParams)

          -- Add up gradParams to params and synchronize back to this thread.
          onmt.utils.Parallel.updateAndSync(self.params[1], _G.gradParams, _G.params, gradBuffer, masterGPU, gmutexId)
        end
      end,
      function(batches, losses, indvAvgLosses, profile)
        if batches then
          iter = iter + #batches
          for i = 1, #batches do
            epochState:update(self.model, batches[i], losses[i])
            if data.needIndividualLosses and data:needIndividualLosses() then
              data:setLoss(getBatchIdx(i), indvAvgLosses[getBatchIdx(i)])
            end
          end
          epochProfiler:add(profile)
        end
      end)

      if iter % self.args.report_every == 0 then
        epochState:log()
        needLog = false
      end
      if iter % self.args.save_every == 0 then
        self.saver:saveIteration(iter, epochState, batchOrder, true)
      end
    end
  end

  if needLog then
    epochState:log(numIterations)
  end

  epochProfiler:stop('train')

  if self.args.profiler then
    _G.logger:info('profile: %s', epochProfiler:log())
  end

  return epochState
end

function Trainer:generateBatchOrder(data, epoch, trainStates)
  local startIteration = 1
  local batchOrder = nil

  if trainStates then
    startIteration = self.args.start_iteration
    batchOrder = trainStates.batchOrder
  elseif epoch > self.args.curriculum then
    batchOrder = torch.randperm(data:batchCount())
  end

  return batchOrder, startIteration
end

function Trainer:train(trainData, validData, trainStates)
  -- Restore previous training states.
  if trainStates then
    if trainStates.rngStates then
      onmt.utils.Cuda.setRNGStates(trainStates.rngStates, true)
    end
    if trainStates.optimStates then
      self.optim:setOptimStates(trainStates.optimStates)
    end
  end

  _G.logger:info('Start training...')

  for epoch = self.args.start_epoch, self.args.end_epoch do
    _G.logger:info('')

    local batchOrder, startIteration = self:generateBatchOrder(trainData, epoch, trainStates)
    local epochState = self:trainEpoch(trainData, epoch, startIteration, batchOrder)
    local validPpl = self:eval(validData)

    _G.logger:info('Validation perplexity: %.2f', validPpl)

    self.optim:updateLearningRate(validPpl, epoch)
    self.saver:saveEpoch(validPpl, epochState, true)

    if self.optim:isFinished() then
      _G.logger:warning('Stopping training due to a too small learning rate value.')
      break
    end

    trainStates = nil
  end
end

return Trainer
