local Memory = {}

--[[ Optimize memory usage of Neural Machine Translation.

Parameters:
  * `model` - a table containing encoder and decoder
  * `criterion` - a single target criterion object
  * `batch` - a Batch object
  * `verbose` - produce output or not

Example:

  local model = {}
  model.encoder = onmt.Models.buildEncoder(...)
  model.decoder = onmt.Models.buildDecoder(...)
  Memory.optimize(model, criterion, batch, verbose)

]]
function Memory.optimize(model, criterion, batch, verbose)

  if verbose then
    _G.logger:info('Preparing memory optimization...')
  end

  -- Prepare memory optimization
  local memoryOptimizer = onmt.utils.MemoryOptimizer.new(model.models)

  -- Batch of one single word since we optimize the first clone.
  local realSizes = { sourceLength = batch.sourceLength, targetLength = batch.targetLength }

  batch.sourceLength = 1
  batch.targetLength = 1

  model:trainNetwork(batch, criterion, true)

  -- mark shared tensors
  local sharedSize, totSize = memoryOptimizer:optimize()

  if verbose then
    _G.logger:info(' * sharing %d%% of output/gradInput tensors memory between clones', (sharedSize / totSize)*100)
  end

  -- Restore batch to be transparent for the calling code.
  batch.sourceLength = realSizes.sourceLength
  batch.targetLength = realSizes.targetLength
end

function Memory.declareOpts(cmd)
  cmd:option('-disable_mem_optimization', false, [[Disable sharing internal of internal buffers between clones - which is in general safe,
                                                   except if you want to look inside clones for visualization purpose for instance.]])
end

return Memory
