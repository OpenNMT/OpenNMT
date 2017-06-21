local Memory = {}

local options = {
  {
    '-disable_mem_optimization', false,
    [[Disable sharing of internal buffers between clones for visualization or development.]]
  }
}

function Memory.declareOpts(cmd)
  cmd:setCmdLineOptions(options)
end

--[[ Optimize memory usage of Neural Machine Translation.

Parameters:
  * `model` - a table containing encoder and decoder
  * `batch` - a Batch object

Example:

  local model = {}
  model.encoder = onmt.Models.buildEncoder(...)
  model.decoder = onmt.Models.buildDecoder(...)
  Memory.optimize(model, batch)

]]
function Memory.optimize(model, batch)

  _G.logger:info('Preparing memory optimization...')

  -- Prepare memory optimization
  local memoryOptimizer = onmt.utils.MemoryOptimizer.new(model.models)

  batch = onmt.utils.Tensor.deepClone(batch)
  batch:resizeSource(1)
  batch.targetLength = 1

  model:trainNetwork(batch, true)

  -- mark shared tensors
  local sharedSize, totSize = memoryOptimizer:optimize()

  _G.logger:info(' * sharing %d%% of output/gradInput tensors memory between clones',
                 (sharedSize / totSize) * 100)
end

return Memory
