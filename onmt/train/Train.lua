local Train = {}
local opt = {}

function Train.declareOpts(cmd)
  cmd:text("")
  cmd:text("**Training options**")
  cmd:text("")
  cmd:option('-save_model', '', [[Model filename (the model will be saved as
                              <save_model>_epochN_PPL.t7 where PPL is the validation perplexity]])
  cmd:option('-epochs', 13, [[Number of training epochs]])
  cmd:option('-start_epoch', 1, [[If loading from a checkpoint, the epoch from which to start]])
  cmd:option('-start_iteration', 1, [[If loading from a checkpoint, the iteration from which to start]])
  cmd:option('-param_init', 0.1, [[Parameters are initialized over uniform distribution with support (-param_init, param_init)]])
  cmd:option('-disable_mem_optimization', false, [[Disable sharing internal of internal buffers between clones - which is in general safe,
                                                except if you want to look inside clones for visualization purpose for instance.]])
  cmd:text("")
  cmd:text("**Logging options**")
  cmd:text("")
  -- bookkeeping
  cmd:option('-save_every', 0, [[Save intermediate models every this many iterations within an epoch.
                             If = 0, will not save models within an epoch. ]])
  cmd:option('-report_every', 50, [[Print stats every this many iterations within an epoch.]])
  -- Optim options.
  onmt.train.Optim.declareOpts(cmd)
end

function Train.init(args)
  opt = args
end

function Train.initParams(model, verbose)
  local numParams = 0
  local params = {}
  local gradParams = {}

  if verbose then
    print('Initializing parameters...')
  end

  -- Order the model table because we need all replicas to have the same order.
  local orderedIndex = {}
  for key in pairs(model) do
    table.insert(orderedIndex, key)
  end
  table.sort(orderedIndex)

  for _, key in ipairs(orderedIndex) do
    local mod = model[key]
    local p, gp = mod:getParameters()

    -- Todo: remove this check.
    if opt.train_from:len() == 0 then
      p:uniform(-opt.param_init, opt.param_init)

      mod:apply(function (m)
        if m.postParametersInitialization then
          m:postParametersInitialization()
        end
      end)
    end

    numParams = numParams + p:size(1)
    table.insert(params, p)
    table.insert(gradParams, gp)
  end
  if verbose then
    print(" * number of parameters: " .. numParams)
  end
  return params, gradParams
end

function Train.eval(model, criterion, data)
  local loss = 0
  local total = 0

  model.encoder:evaluate()
  model.decoder:evaluate()

  for i = 1, data:batchCount() do
    local batch = onmt.utils.Cuda.convert(data:getBatch(i))
    local encoderStates, context = model.encoder:forward(batch)
    loss = loss + model.decoder:computeLoss(batch, encoderStates, context, criterion)
    total = total + batch.targetNonZeros
  end

  model.encoder:training()
  model.decoder:training()

  return math.exp(loss / total)
end


function Train.trainModel(model, trainData, validData, dataset, info, criterionBuilder)
  local params, gradParams = {}, {}
  local criterion

  onmt.utils.Parallel.launch(function(idx)
    -- Only logs information of the first thread.
    local verbose = idx == 1 and not opt.json_log

    _G.params, _G.gradParams = Train.initParams(_G.model, verbose)
    for _, mod in pairs(_G.model) do
      mod:training()
    end

    -- define criterion of each GPU
    _G.criterion = onmt.utils.Cuda.convert(criterionBuilder(dataset))

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

  local optim = onmt.train.Optim.make(#params[1], opt)
  local checkpoint = onmt.train.Checkpoint.new(opt, model, optim, dataset)
  local validPpl = 0

  if not opt.json_log then
    print('Start training...')
  end

  -- TODO: move this out to be its own function.xo
  local function trainEpoch(epoch, lastValidPpl)
    local epochState
    local batchOrder
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

    -- One train iteration.
    local function trainNetwork()
      optim:zeroGrad(_G.gradParams)
      local encStates, context = _G.model.encoder:forward(_G.batch)
      local decOutputs = _G.model.decoder:forward(_G.batch, encStates, context)
      local encGradStatesOut, gradContext, loss = _G.model.decoder:backward(_G.batch, decOutputs, _G.criterion)
      _G.model.encoder:backward(_G.batch, encGradStatesOut, gradContext)
      return loss
    end
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
            _G.batch = batches[idx]
            if _G.batch == nil then
              return idx, 0
            end

            -- Send batch data to the GPU.
            onmt.utils.Cuda.convert(_G.batch)
            _G.batch.totalSize = totalSize
            local loss = trainNetwork()
            return idx, loss
          end,
          function(idx, loss)
            losses[idx]=loss
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
      -- Asynchronous parallel.
      local counter = onmt.utils.Parallel.getCounter()
      counter:set(startI)
      local masterGPU = onmt.utils.Parallel.gpus[1]
      local gradBuffer = onmt.utils.Parallel.gradBuffer
      local gmutexId = onmt.utils.Parallel.gmutexId()

      while counter:get() <= trainData:batchCount() do
        local startCounter = counter:get()
        onmt.utils.Parallel.launch(function(idx)
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
                return lossThread, batchThread
              end
              local i = counter:inc()
              if i > trainData:batchCount() then
                return lossThread, batchThread
              end
              local batchIdx = batchOrder[i]
              if epoch <= opt.curriculum then
                batchIdx = i
              end
              _G.batch = trainData:getBatch(batchIdx)

              -- Send batch data to the GPU.
              onmt.utils.Cuda.convert(_G.batch)
              _G.batch.totalSize = _G.batch.size
              local loss = trainNetwork()

              -- Update the parameters.
              optim:prepareGrad(_G.gradParams, opt.max_grad_norm)

              -- Add up gradParams to params and synchronize back to this thread.
              onmt.utils.Parallel.updateAndSync(params[1], _G.gradParams, _G.params, gradBuffer, masterGPU, gmutexId)

              -- TODO: remove MT specific information.
              batchThread.sourceLength = batchThread.sourceLength + _G.batch.sourceLength * _G.batch.size
              batchThread.targetLength = batchThread.targetLength + _G.batch.targetLength * _G.batch.size
              batchThread.targetNonZeros = batchThread.targetNonZeros + _G.batch.targetNonZeros
              lossThread = lossThread + loss
            end
                                   end,
          function(theloss, thebatch)
            if theloss then
              epochState:update(thebatch, theloss)
            end
        end)
        if opt.report_every > 0 then
          epochState:log(counter:get(), opt.json_log)
        end
        if opt.save_every > 0 then
          checkpoint:saveIteration(counter:get(), epochState, batchOrder, not opt.json_log)
        end
      end
    end
    return epochState
  end


  for epoch = opt.start_epoch, opt.epochs do
    if not opt.json_log then
      print('')
    end
    local epochState = trainEpoch(epoch, validPpl)
    validPpl = Train.eval(model, criterion, validData)
    if not opt.json_log then
      print('Validation perplexity: ' .. validPpl)
    end
    if opt.optim == 'sgd' then
      optim:updateLearningRate(validPpl, epoch)
    end
    checkpoint:saveEpoch(validPpl, epochState, not opt.json_log)
  end
end

return Train
