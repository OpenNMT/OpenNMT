require('onmt.init')
require('onmt.models.seq2seq')

require('tds')
local cmd = onmt.ExtendedCmdLine.new("train.lua")

-------------- Options declaration
local data_options = {
  {'-data',       '', [[Path to the training *-train.t7 file from preprocess.lua]],
                      {valid=onmt.ExtendedCmdLine.nonEmpty}},
  {'-save_model', '', [[Model filename (the model will be saved as
                            <save_model>_epochN_PPL.t7 where PPL is the validation perplexity]],
                      {valid=onmt.ExtendedCmdLine.nonEmpty}}
}

cmd:setCmdLineOptions(data_options, "Data")

-- Generic Model options.
onmt.Model.declareOpts(cmd)

-- Seq2Seq attn options.
onmt.Models.seq2seq.declareOpts(cmd)

-- Optimization options.
onmt.train.Optim.declareOpts(cmd)

-- Training process options.
onmt.Trainer.declareOpts(cmd)

-- Checkpoints options.
onmt.train.Checkpoint.declareOpts(cmd)

cmd:text("")
cmd:text("**Other options**")
cmd:text("")

-- GPU
onmt.utils.Cuda.declareOpts(cmd)
-- Memory optimization
onmt.utils.Memory.declareOpts(cmd)
-- Misc
cmd:option('-seed', 3435, [[Seed for random initialization]], {valid=onmt.ExtendedCmdLine.isUInt()})
-- Logger options
onmt.utils.Logger.declareOpts(cmd)
-- Profiler options
onmt.utils.Profiler.declareOpts(cmd)

local opt = cmd:parse(arg)

local function main()

  torch.manualSeed(opt.seed)

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)
  _G.profiler = onmt.utils.Profiler.new(false)

  onmt.utils.Cuda.init(opt)
  onmt.utils.Parallel.init(opt)

  local checkpoint
  checkpoint, opt = onmt.train.Checkpoint.loadFromCheckpoint(opt)

  -- Create the data loader class.
  _G.logger:info('Loading data from \'' .. opt.data .. '\'...')

  local dataset = torch.load(opt.data, 'binary', false)

  local trainData = onmt.data.Dataset.new(dataset.train.src, dataset.train.tgt)
  local validData = onmt.data.Dataset.new(dataset.valid.src, dataset.valid.tgt)

  trainData:setBatchSize(opt.max_batch_size)
  validData:setBatchSize(opt.max_batch_size)

  _G.logger:info(' * vocabulary size: source = %d; target = %d',
                 dataset.dicts.src.words:size(), dataset.dicts.tgt.words:size())
  _G.logger:info(' * additional features: source = %d; target = %d',
                 #dataset.dicts.src.features, #dataset.dicts.tgt.features)
  _G.logger:info(' * maximum sequence length: source = %d; target = %d',
                 trainData.maxSourceLength, trainData.maxTargetLength)
  _G.logger:info(' * number of training sentences: %d', #trainData.src)
  _G.logger:info(' * maximum batch size: %d', opt.max_batch_size)

  _G.logger:info('Building model...')

  -- main model
  local model

  -- build or load model from checkpoint and copy to GPUs
  onmt.utils.Parallel.launch(function(idx)
    if checkpoint.models then
      _G.model = onmt.Models.seq2seq.new(opt, checkpoint, idx > 1)
    else
      local verbose = idx == 1
      _G.model = onmt.Models.seq2seq.new(opt, dataset, verbose)
    end
    onmt.utils.Cuda.convert(_G.model)
    return idx, _G.model
  end, function(idx, themodel)
    if idx == 1 then
      model = themodel
    end
  end)

  -- Define optimization method.
  local optim = onmt.train.Optim.new(opt, opt.optim_states)
  -- Initialize trainer.
  local trainer = onmt.Trainer.new(opt)

  -- Launch train
  trainer:train(model, optim, trainData, validData, dataset, checkpoint.info)

  -- turn off logger
  _G.logger:shutDown()
end

main()
