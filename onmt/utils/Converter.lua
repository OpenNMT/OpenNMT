---------------------------------------------------------------------------------
-- Local utility functions
---------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------

local Trainer = torch.class('Trainer')

local options = {
  {'-save_every',              0 ,    [[Save intermediate models every this many iterations within an epoch.
                                            If = 0, will not save models within an epoch. ]],
                                      {valid=onmt.utils.ExtendedCmdLine.isUInt()}},
  {'-report_every',            100,    [[Print stats every this many iterations within an epoch.]],
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

function Converter.declareOpts(cmd)
  cmd:setCmdLineOptions(options, 'Converter')
end

function Converter:__init(args)
  self.args = onmt.utils.ExtendedCmdLine.getModuleOpts(args, options)
  -- Use profiler in Converter.
  self.args.profiler = args.profiler
  -- Make a difference with options which is only used in Checkpoint.
  self.options = args
end

function Converter:train(model, optim, trainData, validData, dataset, info)
  --~ local params, gradParams = {}, {}

  --~ onmt.utils.Parallel.launch(function(idx)
    --~ -- Only logs information of the first thread.
    --~ local verbose = idx == 1
--~ 
    --~ -- Initialize and get model parameters.
    --~ _G.params, _G.gradParams = _G.model:initParams(verbose)
--~ 
    --~ -- Switch to training mode.
    --~ _G.model:training()
--~ 
    --~ if self.args.profiler then
      --~ _G.model:enableProfiling()
    --~ end
--~ 
    --~ -- optimize memory of the first clone
    --~ if not self.args.disable_mem_optimization then
      --~ local batch = onmt.utils.Cuda.convert(trainData:getBatch(1))
      --~ batch.totalSize = batch.size
      --~ onmt.utils.Memory.optimize(_G.model, batch, verbose)
    --~ end
--~ 
    --~ return idx, _G.params, _G.gradParams
  --~ end, function(idx, theparams, thegradParams)
    --~ params[idx] = theparams
    --~ gradParams[idx] = thegradParams
  --~ end)
  
  local verbose = true
  
  _G.model:training()
  
  _G.params, _G.gradParams = _G.model:initParams(verbose)
  
end

return Converter
