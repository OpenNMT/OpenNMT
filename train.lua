require('onmt.init')

local path = require('pl.path')
require('tds')
local cmd = torch.CmdLine()

cmd:text("")
cmd:text("**train.lua**")
cmd:text("")

cmd:option('-config', '', [[Read options from this file]])

cmd:text("")
cmd:text("**Data options**")
cmd:text("")

cmd:option('-data', '', [[Path to the training *-train.t7 file from preprocess.lua]])
cmd:option('-save_model', '', [[Model filename (the model will be saved as
                              <save_model>_epochN_PPL.t7 where PPL is the validation perplexity]])
cmd:option('-continue', false, [[If training from a checkpoint, whether to continue the training in the same configuration or not.]])
-- Generic Model options.
onmt.Model.declareOpts(cmd)

cmd:text("")
cmd:text("**Model options**")
cmd:text("")

onmt.Models.seq2seq.declareOpts(cmd)

cmd:text("")
cmd:text("**Optimization options**")
cmd:text("")

onmt.train.Optim.declareOpts(cmd)

cmd:text("")
cmd:text("**Other options**")
cmd:text("")

-- Actual training process option
onmt.Trainer.declareOpts(cmd)
-- GPU
onmt.utils.Cuda.declareOpts(cmd)
-- Memory optimization
onmt.utils.Memory.declareOpts(cmd)
-- Misc
cmd:option('-no_nccl', false, [[Disable usage of nccl in parallel mode.]])
cmd:option('-seed', 3435, [[Seed for random initialization]])

onmt.utils.Logger.declareOpts(cmd)
onmt.utils.Profiler.declareOpts(cmd)

local opt = cmd:parse(arg)

local function main()
  local requiredOptions = {
    "data",
    "save_model"
  }

  onmt.utils.Opt.init(opt, requiredOptions)

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)
  _G.profiler = onmt.utils.Profiler.new(false)

  onmt.utils.Cuda.init(opt)
  onmt.utils.Parallel.init(opt)

  local checkpoint = {}

  if opt.train_from:len() > 0 then
    assert(path.exists(opt.train_from), 'checkpoint path invalid')

    if not opt.json_log then
      _G.logger:info('Loading checkpoint \'' .. opt.train_from .. '\'...')
    end

    checkpoint = torch.load(opt.train_from)

    opt.layers = checkpoint.options.layers
    opt.rnn_size = checkpoint.options.rnn_size
    opt.brnn = checkpoint.options.brnn
    opt.brnn_merge = checkpoint.options.brnn_merge
    opt.input_feed = checkpoint.options.input_feed

    -- Resume training from checkpoint
    if opt.continue then
      opt.optim = checkpoint.options.optim
      opt.learning_rate_decay = checkpoint.options.learning_rate_decay
      opt.start_decay_at = checkpoint.options.start_decay_at
      opt.curriculum = checkpoint.options.curriculum

      opt.learning_rate = checkpoint.info.learningRate
      opt.optim_states = checkpoint.info.optimStates
      opt.start_epoch = checkpoint.info.epoch
      opt.start_iteration = checkpoint.info.iteration

      if not opt.json_log then
        _G.logger:info('Resuming training from epoch ' .. opt.start_epoch
                         .. ' at iteration ' .. opt.start_iteration .. '...')
      end
    end
  end

  -- Create the data loader class.
  if not opt.json_log then
    _G.logger:info('Loading data from \'' .. opt.data .. '\'...')
  end

  local dataset = torch.load(opt.data, 'binary', false)

  local trainData = onmt.data.Dataset.new(dataset.train.src, dataset.train.tgt)
  local validData = onmt.data.Dataset.new(dataset.valid.src, dataset.valid.tgt)

  trainData:setBatchSize(opt.max_batch_size)
  validData:setBatchSize(opt.max_batch_size)

  if not opt.json_log then
    _G.logger:info(' * vocabulary size: source = %d; target = %d',
                   dataset.dicts.src.words:size(), dataset.dicts.tgt.words:size())
    _G.logger:info(' * additional features: source = %d; target = %d',
                   #dataset.dicts.src.features, #dataset.dicts.tgt.features)
    _G.logger:info(' * maximum sequence length: source = %d; target = %d',
                   trainData.maxSourceLength, trainData.maxTargetLength)
    _G.logger:info(' * number of training sentences: %d', #trainData.src)
    _G.logger:info(' * maximum batch size: %d', opt.max_batch_size)
  else
    local metadata = {
      options = opt,
      vocabSize = {
        source = dataset.dicts.src.words:size(),
        target = dataset.dicts.tgt.words:size()
      },
      additionalFeatures = {
        source = #dataset.dicts.src.features,
        target = #dataset.dicts.tgt.features
      },
      sequenceLength = {
        source = trainData.maxSourceLength,
        target = trainData.maxTargetLength
      },
      trainingSentences = #trainData.src
    }

    onmt.utils.Log.logJson(metadata)
  end

  if not opt.json_log then
    _G.logger:info('Building model...')
  end

  local model

  onmt.utils.Parallel.launch(function(idx)

    if checkpoint.models then
      _G.model = onmt.Models.seq2seq.new(opt, checkpoint, idx > 1)
    else
      local verbose = idx == 1 and not opt.json_log
      _G.model = onmt.Models.seq2seq.new(opt, dataset, verbose)
    end

    onmt.utils.Cuda.convert(_G.model)

    return idx, _G.model
  end, function(idx, themodel)
    if idx == 1 then
      model = themodel
    end
  end)

  local optim = onmt.train.Optim.new({
    method = opt.optim,
    learningRate = opt.learning_rate,
    learningRateDecay = opt.learning_rate_decay,
    startDecayAt = opt.start_decay_at,
    optimStates = opt.optim_states,
    max_grad_norm = opt.max_grad_norm
  })

  local trainer = onmt.Trainer.new(opt)

  trainer:train(model, optim, trainData, validData, dataset, checkpoint.info)

  _G.logger:shutDown()
end

main()
