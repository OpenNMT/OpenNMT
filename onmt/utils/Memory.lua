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
  * `verbose` - produce output or not

Example:

  local model = {}
  model.encoder = onmt.Models.buildEncoder(...)
  model.decoder = onmt.Models.buildDecoder(...)
  Memory.optimize(model, batch, verbose)

]]
function Memory.optimize(model, batch, verbose)

  if verbose then
    _G.logger:info('Preparing memory optimization...')
  end

  -- Prepare memory optimization
  local memoryOptimizer = onmt.utils.MemoryOptimizer.new(model.models)

  batch = onmt.utils.Tensor.deepClone(batch)
  batch.sourceLength = 1
  batch.targetLength = 1
  batch.uneven = false

  model:trainNetwork(batch, true)

  -- mark shared tensors
  local sharedSize, totSize = memoryOptimizer:optimize()

  if verbose then
    _G.logger:info(' * sharing %d%% of output/gradInput tensors memory between clones', (sharedSize / totSize)*100)
  end
end

return Memory
