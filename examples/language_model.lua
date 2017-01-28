require 'onmt.init'

local tds = require('tds')

local cmd = torch.CmdLine()

cmd:text("")
cmd:text("**language-model.lua**")
cmd:text("")

cmd:option('-config', '', [[Read options from this file]])

cmd:text("")
cmd:text("**Data options**")
cmd:text("")

cmd:option('-train', '', [[Path to the training data]])
cmd:option('-valid', '', [[Path to the validation  data]])
cmd:option('-vocab_size', 50000, [[Size of the source vocabulary]])
cmd:option('-vocab', '', [[Path to an existing source vocabulary]])
cmd:option('-features_vocabs_prefix', '', [[Path prefix to existing features vocabularies]])

cmd:option('-seq_length', 50, [[Maximum source sequence length]])
cmd:option('-shuffle', 1, [[Shuffle data]])
cmd:option('-seed', 3435, [[Random seed]])

cmd:text("")
cmd:text("**Model options**")
cmd:text("")

cmd:option('-save_model', '', [[Model filename (the model will be saved as
                              <save_model>_epochN_PPL.t7 where PPL is the validation perplexity]])

onmt.Models.lm.declareOpts(cmd)

cmd:text("")
cmd:text("**Optimization options**")
cmd:text("")

onmt.train.Optim.declareOpts(cmd)

cmd:text("")
cmd:text("**Other options**")
cmd:text("")

-- GPU
onmt.utils.Cuda.declareOpts(cmd)
cmd:option('-async_parallel', false, [[Use asynchronous parallelism training.]])
cmd:option('-async_parallel_minbatch', 1000, [[For async parallel computing, minimal number of batches before being parallel.]])
cmd:option('-no_nccl', false, [[Disable usage of nccl in parallel mode.]])

-- bookkeeping
cmd:option('-report_every', 50, [[Print stats every this many iterations within an epoch.]])
cmd:option('-seed', 3435, [[Seed for random initialization]])

onmt.utils.Logger.declareOpts(cmd)

local opt = cmd:parse(arg)

local function isValid(sent, maxSeqLength)
  return #sent > 0 and #sent <= maxSeqLength
end

local function vecToTensor(vec)
  local t = torch.Tensor(#vec)
  for i, v in pairs(vec) do
    t[i] = v
  end
  return t
end

local function makeData(file, dicts)
  local dataset = tds.Vec()
  local features = tds.Vec()

  local sizes = tds.Vec()

  local count = 0
  local ignored = 0

  local reader = onmt.utils.FileReader.new(file)

  while true do
    local tokens = reader:next()

    if tokens == nil then
      break
    end

    if isValid(tokens, opt.seq_length) then
      local words, feats = onmt.utils.Features.extract(tokens)

      dataset:insert(dicts.words:convertToIdx(words, onmt.Constants.UNK_WORD))

      if #dicts.features > 0 then
        features:insert(onmt.utils.Features.generateSource(dicts.features, feats, true))
      end

      sizes:insert(#words)
    else
      ignored = ignored + 1
    end

    count = count + 1

  end

  reader:close()

  local function reorderData(perm)
    dataset = onmt.utils.Table.reorder(dataset, perm, true)

    if #dicts.features > 0 then
      features = onmt.utils.Table.reorder(features, perm, true)
    end
  end

  if opt.shuffle == 1 then
    _G.logger:info('... shuffling sentences')
    local perm = torch.randperm(#dataset)
    sizes = onmt.utils.Table.reorder(sizes, perm, true)
    reorderData(perm)
  end

  _G.logger:info('... sorting sentences by size')
  local _, perm = torch.sort(vecToTensor(sizes), true)
  reorderData(perm)

  _G.logger:info('Prepared ' .. #dataset .. ' sentences (' .. ignored
                   .. ' ignored due to length > ' .. opt.seq_length .. ')')

  local data = {
    words = dataset,
    features = features
  }

  return data
end

local function main()
  local requiredOptions = {
    "train",
    "valid",
    "save_model"
  }

  onmt.utils.Opt.init(opt, requiredOptions)

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)

  onmt.utils.Cuda.init(opt)

  local Vocabulary = onmt.data.Vocabulary

  local data = {}

  data.dicts = {}
  data.dicts = Vocabulary.init('train', opt.train, opt.vocab, opt.vocab_size,
                               opt.features_vocabs_prefix, function(s) return isValid(s, opt.seq_length) end)

  _G.logger:info('Preparing training data...')
  data.train = makeData(opt.train, data.dicts)
  _G.logger:info('')

  _G.logger:info('Preparing validation data...')
  data.valid = makeData(opt.valid, data.dicts)
  _G.logger:info('')

  local trainData = onmt.data.Dataset.new(data.train, nil)
  local validData = onmt.data.Dataset.new(data.valid, nil)

  trainData:setBatchSize(opt.max_batch_size)
  validData:setBatchSize(opt.max_batch_size)

  _G.logger:info('Building models...')

  onmt.utils.Parallel.launch(function(idx)

    if checkpoint.models then
      _G.model = onmt.Models.lm.new(opt, checkpoint, idx > 1)
    else
      local verbose = idx == 1 and not opt.json_log
      _G.model = onmt.Models.lm.new(opt, dataset, verbose)
    end

    onmt.utils.Cuda.convert(_G.model)

    return idx, _G.model
  end, function(idx, themodel)
    if idx == 1 then
      model = themodel
    end
  end)

  _G.logger:info('')

  local optim = onmt.train.Optim.new({
    method = opt.optim,
    learningRate = opt.learning_rate,
    learningRateDecay = opt.learning_rate_decay,
    startDecayAt = opt.start_decay_at,
    optimStates = opt.optim_states
  })

  onmt.Trainer.train(opt, model, optim, trainData, validData, dataset, checkpoint.info)

  _G.logger:shutDown()

end

main()
