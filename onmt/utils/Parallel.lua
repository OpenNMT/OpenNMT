--[[
  This file provides generic parallel class - allowing to run functions
  in different threads and on different GPU
]]--

local Parallel = {
  _pool = nil,
  count = 1,
  gradBuffer = torch.Tensor()
}

-- Synchronizes the current stream on dst device with src device. This is only
-- necessary if we are not on the default stream
local function waitForDevice(dst, src)
  if dst > 0 and src > 0 then
    local stream = cutorch.getStream()
    if stream ~= 0 then
      cutorch.streamWaitForMultiDevice(dst, stream, { [src] = {stream} })
    end
  end
end

function Parallel.getCounter()
  return Parallel._tds.AtomicCounter()
end

function Parallel.gmutexId()
  return Parallel._gmutex:id()
end

function Parallel.init(opt)
  Parallel.count = onmt.utils.Cuda.replicaCount()
  Parallel.gradBuffer = onmt.utils.Cuda.convert(Parallel.gradBuffer)
  Parallel._tds = require('tds')

  if Parallel.count > 1 then
    local globalLogger = _G.logger
    local globalProfiler = _G.profiler
    local threads = require('threads')
    threads.Threads.serialization('threads.sharedserialize')
    Parallel._gmutex = threads.Mutex()
    local deviceIds = onmt.utils.Cuda.deviceIds
    Parallel._pool = threads.Threads(
      Parallel.count,
      function(threadid)
        if deviceIds[threadid] > 0 then
          require('cunn')
          require('nngraph')
        end
        require('onmt.init')
        _G.threads = require('threads')
      end,
      function(threadid)
        _G.logger = globalLogger
        _G.profiler = globalProfiler
        onmt.utils.Cuda.init(opt, deviceIds[threadid])
      end
    ) -- dedicate threads to devices
    Parallel._pool:specific(true)
  end

  if Parallel.count > 1 and #onmt.utils.Cuda.gpuIds > 0 and not opt.no_nccl then
    -- check if we have nccl installed
    local ret
    ret, Parallel.usenccl = pcall(require, 'nccl')
    if not ret then
      _G.logger:warning("For improved efficiency with multiple GPUs, consider installing nccl")
      Parallel.usenccl = nil
    elseif os.getenv('CUDA_LAUNCH_BLOCKING') == '1' then
      _G.logger:warning("CUDA_LAUNCH_BLOCKING set - cannot use nccl")
      Parallel.usenccl = nil
    end
  end
end

--[[ Launch function in parallel on different threads. ]]
function Parallel.launch(closure, endCallback)
  endCallback = endCallback or function() end
  for j = 1, Parallel.count do
    if Parallel._pool == nil then
      endCallback(closure(j))
    else
      Parallel._pool:addjob(j, function() return closure(j) end, endCallback)
    end
  end
  if Parallel._pool then
    Parallel._pool:synchronize()
  end
end

--[[ Accumulate the gradient parameters from the different parallel threads. ]]
function Parallel.accGradParams(gradParams, batches)
  if Parallel.count > 1 then
    for h = 1, #gradParams[1] do
      local inputs = { gradParams[1][h] }
      for j = 2, #batches do
        if not Parallel.usenccl then
          -- TODO - this is memory costly since we need to clone full parameters from one GPU to another
          -- to avoid out-of-memory, we can copy/add by batch

         -- Synchronize before and after copy to ensure that it doesn't overlap
         -- with this add or previous adds
          waitForDevice(onmt.utils.Cuda.deviceIds[j], onmt.utils.Cuda.deviceIds[1])
          local remoteGrads = onmt.utils.Tensor.reuseTensor(Parallel.gradBuffer, gradParams[j][h]:size())
          remoteGrads:copy(gradParams[j][h])
          waitForDevice(onmt.utils.Cuda.deviceIds[1], onmt.utils.Cuda.deviceIds[j])
          gradParams[1][h]:add(remoteGrads)
        else
          table.insert(inputs, gradParams[j][h])
        end
      end
      if Parallel.usenccl then
        Parallel.usenccl.reduce(inputs, nil, true)
      end
    end
  end
end

-- [[ In async mode, sync the parameters from all replica to master replica. ]]
function Parallel.updateAndSync(masterParams, replicaGradParams, replicaParams, gradBuffer, masterGPU, gmutexId)
  -- Add a mutex to avoid competition while accessing shared buffer and while updating parameters.
  local mutex = _G.threads.Mutex(gmutexId)
  mutex:lock()
  local device = cutorch.getDevice()
  cutorch.setDevice(masterGPU)
  for h = 1, #replicaGradParams do
    waitForDevice(device, masterGPU)
    local remoteGrads = onmt.utils.Tensor.reuseTensor(gradBuffer, replicaGradParams[h]:size())
    remoteGrads:copy(replicaGradParams[h])
    waitForDevice(masterGPU, device)
    masterParams[h]:add(remoteGrads)
  end
  cutorch.setDevice(device)
  for h = 1, #replicaGradParams do
    replicaParams[h]:copy(masterParams[h])
    waitForDevice(device, masterGPU)
  end
  mutex:unlock()
end

--[[ Sync parameters from main model to different parallel threads. ]]
function Parallel.syncParams(params)
  if Parallel.count > 1 then
    if not Parallel.usenccl then
      for j = 2, Parallel.count do
        for h = 1, #params[1] do
          params[j][h]:copy(params[1][h])
        end
        waitForDevice(onmt.utils.Cuda.deviceIds[j], onmt.utils.Cuda.deviceIds[1])
      end
    else
      for h = 1, #params[1] do
        local inputs = { params[1][h] }
        for j = 2, Parallel.count do
          table.insert(inputs, params[j][h])
        end
        Parallel.usenccl.bcast(inputs, true, 1)
      end
    end
  end
end

return Parallel
