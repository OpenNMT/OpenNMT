require 'onmt.init'

local tds = require('tds')

--[[ command line arguments ]]--
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

cmd:option('-layers', 2, [[Number of layers in the RNN encoder/decoder]])
cmd:option('-rnn_size', 500, [[Size of RNN hidden states]])
cmd:option('-rnn_type', 'LSTM', [[Type of RNN cell: LSTM, GRU]])
cmd:option('-word_vec_size', 500, [[Word embedding sizes]])
cmd:option('-feat_merge', 'concat', [[Merge action for the features embeddings: concat or sum]])
cmd:option('-feat_vec_exponent', 0.7, [[When using concatenation, if the feature takes N values
                                      then the embedding dimension will be set to N^exponent]])
cmd:option('-feat_vec_size', 20, [[When using sum, the common embedding size of the features]])
cmd:option('-residual', false, [[Add residual connections between RNN layers.]])
cmd:option('-brnn', false, [[Use a bidirectional encoder]])
cmd:option('-brnn_merge', 'sum', [[Merge action for the bidirectional hidden states: concat or sum]])

cmd:text("")
cmd:text("**Optimization options**")
cmd:text("")

cmd:option('-max_batch_size', 64, [[Maximum batch size]])
cmd:option('-end_epoch', 13, [[The final epoch of the training]])
cmd:option('-param_init', 0.1, [[Parameters are initialized over uniform distribution with support (-param_init, param_init)]])
cmd:option('-optim', 'sgd', [[Optimization method. Possible options are: sgd, adagrad, adadelta, adam]])
cmd:option('-learning_rate', 1, [[Starting learning rate. If adagrad/adadelta/adam is used,
                                then this is the global learning rate. Recommended settings are: sgd = 1,
                                adagrad = 0.1, adadelta = 1, adam = 0.0002]])
cmd:option('-max_grad_norm', 5, [[If the norm of the gradient vector exceeds this renormalize it to have the norm equal to max_grad_norm]])
cmd:option('-dropout', 0.3, [[Dropout probability. Dropout is applied between vertical LSTM stacks.]])
cmd:option('-learning_rate_decay', 0.5, [[Decay learning rate by this much if (i) perplexity does not decrease
                                        on the validation set or (ii) epoch has gone past the start_decay_at_limit]])
cmd:option('-start_decay_at', 9, [[Start decay after this epoch]])
cmd:option('-pre_word_vecs_enc', '', [[If a valid path is specified, then this will load
                                     pretrained word embeddings.
                                     See README for specific formatting instructions.]])
cmd:option('-fix_word_vecs_enc', false, [[Fix word embeddings]])

cmd:text("")
cmd:text("**Other options**")
cmd:text("")

-- GPU
onmt.utils.Cuda.declareOpts(cmd)
cmd:option('-async_parallel', false, [[Use asynchronous parallelism training.]])
cmd:option('-async_parallel_minbatch', 1000, [[For async parallel computing, minimal number of batches before being parallel.]])
cmd:option('-no_nccl', false, [[Disable usage of nccl in parallel mode.]])
cmd:option('-disable_mem_optimization', false, [[Disable sharing internal of internal buffers between clones - which is in general safe,
                                                except if you want to look inside clones for visualization purpose for instance.]])

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

local function initParams(model, verbose)
  local numParams = 0
  local params = {}
  local gradParams = {}

  if verbose then
    _G.logger:info('Initializing parameters...')
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
    p:uniform(-opt.param_init, opt.param_init)

    mod:apply(function (m)
     if m.postParametersInitialization then
       m:postParametersInitialization()
     end
    end)

    numParams = numParams + p:size(1)
    table.insert(params, p)
    table.insert(gradParams, gp)
  end

  if verbose then
    _G.logger:info(" * number of parameters: " .. numParams)
  end

  return params, gradParams
end

local function buildCriterion(vocabSize, features)
  local criterion = nn.ParallelCriterion(false)

  local function addNllCriterion(size)
    -- Ignores padding value.
    local w = torch.ones(size)
    w[onmt.Constants.PAD] = 0

    local nll = nn.ClassNLLCriterion(w)

    -- Let the training code manage loss normalization.
    nll.sizeAverage = false
    criterion:add(nll)
  end

  addNllCriterion(vocabSize)

  for j = 1, #features do
    addNllCriterion(features[j]:size())
  end

  return criterion
end

local function eval(model, criterion, data)
  local loss = 0
  local total = 0

  model.encoder:evaluate()
  model.generator:evaluate()

  for i = 1, data:batchCount() do
    local batch = onmt.utils.Cuda.convert(data:getBatch(t))
    local _, context = model.encoder:forward(batch)
    local EOS_vector = torch.LongTensor(batch.size):fill(dicts.words:lookup(onmt.Constants.EOS_WORD))
    onmt.utils.Cuda.convert(EOS_vector)

    for t = 1, batch.sourceLength do
      local genOutputs = model.generator:forward(context:select(2,t))
      -- LM is supposed to predict following word
      local output
      if t ~= batch.sourceLength then
        output = batch:getSourceInput(t+1)
      else
        output = EOS_vector
      end
      -- same format with and without features
      if torch.type(output) ~= 'table' then output = { output } end
      loss = loss + criterion:forward(genOutputs, output)
    end

    total = total + batch.sourceLength*batch.size

  end

  model.encoder:training()
  model.generator:training()

  return math.exp(loss / total)
end

local function trainModel(model, trainData, validData, dicts)
  local params, gradParams
  local criterion

  params, gradParams = initParams(model, true)
  for _, mod in pairs(model) do
    mod:training()
  end

  -- define criterion
  criterion = onmt.utils.Cuda.convert(buildCriterion(dicts.words:size(), dicts.features))

  local optim = onmt.train.Optim.new({
    method = opt.optim,
    numModels = #params,
    learningRate = opt.learning_rate,
    learningRateDecay = opt.learning_rate_decay,
    startDecayAt = opt.start_decay_at,
    optimStates = opt.optim_states
  })

  local EOS_vector = torch.LongTensor(opt.max_batch_size):fill(dicts.words:lookup(onmt.Constants.EOS_WORD))
  onmt.utils.Cuda.convert(EOS_vector)

  local function trainEpoch(epoch, lastValidPpl)
    local numIterations = trainData:batchCount()
    local epochState = onmt.train.EpochState.new(epoch, numIterations, optim:getLearningRate(), lastValidPpl)
    -- Shuffle mini batch order.
    local batchOrder = torch.randperm(trainData:batchCount())

    local function trainNetwork(batch)
      optim:zeroGrad(gradParams)
      local loss = 0
      local _, context = model.encoder:forward(batch)
      local gradContexts = torch.Tensor(batch.size, batch.sourceLength, opt.rnn_size)
      gradContexts = onmt.utils.Cuda.convert(gradContexts)
      -- for each word of the sentence, generate target
      for t = 1, batch.sourceLength do
        local genOutputs = model.generator:forward(context:select(2,t))
        -- LM is supposed to predict following word
        local output
        if t ~= batch.sourceLength then
          output = batch:getSourceInput(t+1)
        else
          output = EOS_vector:narrow(1,1,batch.size)
        end
        -- same format with and without features
        if torch.type(output) ~= 'table' then output = { output } end
        loss = loss + criterion:forward(genOutputs, output)
        -- backward
        local genGradOutput = criterion:backward(genOutputs, output)
        for j = 1, #genGradOutput do
          genGradOutput[j]:div(batch.totalSize)
        end
        gradContexts[{{},t}]:copy(model.generator:backward(context:select(2,t), genGradOutput))
      end
      model.encoder:backward(batch, nil, gradContexts)
      return loss
    end

    local iter = 1
    for i = 1, trainData:batchCount() do
      local batchIdx = batchOrder[i]
      local batch = trainData:getBatch(batchIdx)
      -- Send batch data to the GPU.
      onmt.utils.Cuda.convert(batch)
      batch.totalSize = batch.size

      local loss = trainNetwork(batch)

      -- Update the parameters.
      optim:prepareGrad(gradParams, opt.max_grad_norm)
      optim:updateParams(params, gradParams)

      epochState:update(batch, loss)
      if iter % opt.report_every == 0 then
        epochState:log(iter, opt.json_log)
      end
      iter = iter + 1
    end

    return epochState
  end


  local validPpl = 0

  _G.logger:info('Start training...')

  for epoch = 1, opt.end_epoch do
    _G.logger:info('')

    trainEpoch(epoch, validPpl)

    validPpl = eval(model, criterion, validData)

    _G.logger:info('Validation perplexity: %.2f', validPpl)

    if opt.optim == 'sgd' then
      optim:updateLearningRate(validPpl, epoch)
    end
  end

end

local function main()
  local requiredOptions = {
    "train",
    "valid"
  }

  onmt.utils.Opt.init(opt, requiredOptions)
  onmt.utils.Cuda.init(opt)

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)

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
  local model={}
  model.encoder = onmt.Models.buildEncoder(opt, data.dicts)
  if #data.dicts.features > 0 then
    model.generator = onmt.FeaturesGenerator.new(opt.rnn_size, data.dicts.words:size(), data.dicts.features)
  else
    model.generator = onmt.Generator.new(opt.rnn_size, data.dicts.words:size())
  end

  for _, mod in pairs(model) do
    onmt.utils.Cuda.convert(mod)
  end
  _G.logger:info('')

  trainModel(model, trainData, validData, data.dicts)

  _G.logger:shutDown()
end

main()
