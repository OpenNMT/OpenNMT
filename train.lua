require('./lib/utils')
require('./lib/train')
require('./lib/onmt')

local path = require('pl.path')

local constants = require('lib.constants')
local Data = require('lib.data')
local Models = require('lib.models')

local cmd = torch.CmdLine()

cmd:text("")
cmd:text("**train.lua**")
cmd:text("")

cmd:option('-config', '', [[Read options from this file]])

cmd:text("")
cmd:text("**Data options**")
cmd:text("")

cmd:option('-data', '', [[Path to the training *-train.t7 file from preprocess.lua]])
cmd:option('-save_file', '', [[Savefile name (model will be saved as
                                             savefile_epochX_PPL.t7 where X is the X-th epoch and PPL is
                                             the validation perplexity]])
cmd:option('-train_from', '', [[If training from a checkpoint then this is the path to the pretrained model.]])
cmd:option('-continue', false, [[If training from a checkpoint, whether to continue the training in the same configuration or not.]])

cmd:text("")
cmd:text("**Model options**")
cmd:text("")

cmd:option('-num_layers', 2, [[Number of layers in the LSTM encoder/decoder]])
cmd:option('-rnn_size', 500, [[Size of LSTM hidden states]])
cmd:option('-word_vec_size', 500, [[Word embedding sizes]])
cmd:option('-feat_vec_exponent', 0.7, [[If the feature takes N values, then the
                                      embedding dimension will be set to N^exponent]])
cmd:option('-input_feed', 1, [[Feed the context vector at each time step as additional input (via concatenation with the word embeddings) to the decoder.]])
cmd:option('-brnn', false, [[Use a bidirectional encoder]])
cmd:option('-brnn_merge', 'sum', [[Merge action for the bidirectional hidden states: concat or sum]])

cmd:text("")
cmd:text("**Optimization options**")
cmd:text("")

cmd:option('-max_batch_size', 64, [[Maximum batch size]])
cmd:option('-epochs', 13, [[Number of training epochs]])
cmd:option('-start_epoch', 1, [[If loading from a checkpoint, the epoch from which to start]])
cmd:option('-start_iteration', 1, [[If loading from a checkpoint, the iteration from which to start]])
cmd:option('-param_init', 0.1, [[Parameters are initialized over uniform distribution with support (-param_init, param_init)]])
cmd:option('-optim', 'sgd', [[Optimization method. Possible options are: sgd, adagrad, adadelta, adam]])
cmd:option('-learning_rate', 1, [[Starting learning rate. If adagrad/adadelta/adam is used,
                                then this is the global learning rate. Recommended settings: sgd =1,
                                adagrad = 0.1, adadelta = 1, adam = 0.1]])
cmd:option('-max_grad_norm', 5, [[If the norm of the gradient vector exceeds this renormalize it to have the norm equal to max_grad_norm]])
cmd:option('-dropout', 0.3, [[Dropout probability. Dropout is applied between vertical LSTM stacks.]])
cmd:option('-lr_decay', 0.5, [[Decay learning rate by this much if (i) perplexity does not decrease
                             on the validation set or (ii) epoch has gone past the start_decay_at_limit]])
cmd:option('-start_decay_at', 9, [[Start decay after this epoch]])
cmd:option('-curriculum', 0, [[For this many epochs, order the minibatches based on source
                             sequence length. Sometimes setting this to 1 will increase convergence speed.]])
cmd:option('-pre_word_vecs_enc', '', [[If a valid path is specified, then this will load
                                     pretrained word embeddings on the encoder side.
                                     See README for specific formatting instructions.]])
cmd:option('-pre_word_vecs_dec', '', [[If a valid path is specified, then this will load
                                     pretrained word embeddings on the decoder side.
                                     See README for specific formatting instructions.]])
cmd:option('-fix_word_vecs_enc', false, [[Fix word embeddings on the encoder side]])
cmd:option('-fix_word_vecs_dec', false, [[Fix word embeddings on the decoder side]])

cmd:text("")
cmd:text("**Other options**")
cmd:text("")

-- GPU
cmd:option('-gpuid', -1, [[Which gpu to use (1-indexed). < 1 = use CPU]])
cmd:option('-nparallel', 1, [[When using GPUs, how many batches to execute in parallel.
                            Note: this will technically change the final batch size to max_batch_size*nparallel.]])
cmd:option('-no_nccl', false, [[Disable usage of nccl in parallel mode.]])
cmd:option('-disable_mem_optimization', false, [[Disable sharing internal of internal buffers between clones - which is in general safe,
                                                except if you want to look inside clones for visualization purpose for instance.]])

-- bookkeeping
cmd:option('-save_every', 0, [[Save intermediate models every this many iterations within an epoch.
                             If = 0, will not save models within an epoch. ]])
cmd:option('-print_every', 50, [[Print stats every this many iterations within an epoch.]])
cmd:option('-seed', 3435, [[Seed for random initialization]])
cmd:option('-no_log', false, [[By default, a log file save_file.log is created during training giving time, ppl, and free memory at each
                             epoch. Use this flag to disable.]])

local opt = cmd:parse(arg)

local function get_nets(model)
  local nets = {}

  if opt.brnn then
    nets.encoder = model.encoder.fwd.network
    nets.encoder_bwd = model.encoder.bwd.network
  else
    nets.encoder = model.encoder.network
  end

  nets.decoder = model.decoder.network
  nets.generator = model.decoder.generator

  return nets
end

local function init_params(nets, verbose)
  local num_params = 0
  local params = {}
  local grad_params = {}

  if verbose then
    print('Initializing parameters...')
  end

  for _, net in pairs(nets) do
    local p, gp = net:getParameters()

    if opt.train_from:len() == 0 then
      p:uniform(-opt.param_init, opt.param_init)
    end

    num_params = num_params + p:size(1)
    table.insert(params, p)
    table.insert(grad_params, gp)
  end

  if verbose then
    print(" * number of parameters: " .. num_params)
  end

  return params, grad_params
end

local function build_criterion(vocab_size, features)
  local criterion = nn.ParallelCriterion(false)

  local function add_nll_criterion(size)
    -- Ignores padding value.
    local w = torch.ones(size)
    w[constants.PAD] = 0

    local nll = nn.ClassNLLCriterion(w)

    -- Let the training code manage loss normalization.
    nll.sizeAverage = false
    criterion:add(nll)
  end

  add_nll_criterion(vocab_size)

  for j = 1, #features do
    add_nll_criterion(#features[j])
  end

  return criterion
end

local function eval(model, criterion, data)
  local loss = 0
  local total = 0

  model.encoder:evaluate()
  model.decoder:evaluate()

  for i = 1, #data do
    local batch = utils.Cuda.convert(data:get_batch(i))
    local encoder_states, context = model.encoder:forward(batch)
    loss = loss + model.decoder:compute_loss(batch, encoder_states, context, criterion)
    total = total + batch.target_non_zeros
  end

  model.encoder:training()
  model.decoder:training()

  return math.exp(loss / total)
end

local function train_model(model, train_data, valid_data, dataset, info, log)
  local params, grad_params = {}, {}
  local criterion

  local optim = train.Optim.new({
    method = opt.optim,
    num_models = #params,
    learning_rate = opt.learning_rate,
    lr_decay = opt.lr_decay,
    start_decay_at = opt.start_decay_at,
    optim_states = opt.optim_states
  })

  local checkpoint = train.Checkpoint.new(opt, model, optim, dataset)

  utils.Parallel.launch(nil, function(idx)
    -- Only logs information of the first thread.
    local verbose = idx == 1

    local nets = get_nets(_G.model)
    _G.params, _G.grad_params = init_params(nets, verbose)
    for _, mod in pairs(_G.model) do
      mod:training()
    end

    -- define criterion of each GPU
    _G.criterion = utils.Cuda.convert(build_criterion(#dataset.dicts.targ.words,
                                                      dataset.dicts.targ.features))

    -- optimize memory of the first clone
    if not opt.disable_mem_optimization then
      local batch = utils.Cuda.convert(train_data:get_batch(1))
      batch.total_size = batch.size
      utils.Memory.optimize(_G.model, _G.criterion, batch, verbose)
    end

    return idx, _G.criterion, _G.params, _G.grad_params
  end, function(idx, thecriterion, theparams, thegrad_params)
    if idx == 1 then criterion = thecriterion end
    params[idx] = theparams
    grad_params[idx] = thegrad_params
  end)

  local function train_epoch(epoch)
    local epoch_state
    local batch_order

    local start_i = opt.start_iteration
    local num_iterations = math.ceil(#train_data / utils.Parallel.count)

    if start_i > 1 and info ~= nil then
      epoch_state = train.EpochState.new(epoch, num_iterations, optim:get_learning_rate(), info.epoch_status)
      batch_order = info.batch_order
    else
      epoch_state = train.EpochState.new(epoch, num_iterations, optim:get_learning_rate())
      -- shuffle mini batch order
      batch_order = torch.randperm(#train_data)
    end

    opt.start_iteration = 1
    local iter = 1

    for i = start_i, #train_data, utils.Parallel.count do
      local batches = {}
      local total_size = 0

      for j = 1, math.min(utils.Parallel.count, #train_data-i+1) do
        local batch_idx = batch_order[i+j-1]
        if epoch <= opt.curriculum then
          batch_idx = i+j-1
        end
        table.insert(batches, train_data:get_batch(batch_idx))
        total_size = total_size + batches[#batches].size
      end

      local losses = {}

      utils.Parallel.launch(nil, function(idx)
        _G.batch = batches[idx]
        if _G.batch == nil then
          return idx, 0
        end

        -- send batch data to GPU
        utils.Cuda.convert(_G.batch)
        _G.batch.total_size = total_size

        optim:zero_grad(_G.grad_params)

        local enc_states, context = _G.model.encoder:forward(_G.batch)
        local dec_outputs = _G.model.decoder:forward(_G.batch, enc_states, context)

        local enc_grad_states_out, grad_context, loss = _G.model.decoder:backward(_G.batch, dec_outputs, _G.criterion)
        _G.model.encoder:backward(_G.batch, enc_grad_states_out, grad_context)
        return idx, loss
      end,
      function(idx, loss) losses[idx]=loss end)

      -- accumulate the gradients from the different parallel threads
      utils.Parallel.accGradParams(grad_params, batches)

      -- update the parameters
      optim:update_params(params[1], grad_params[1], opt.max_grad_norm)

      -- sync the paramaters with the different parallel threads
      utils.Parallel.syncParams(params)

      epoch_state:update(batches, losses)

      if iter % opt.print_every == 0 then
        epoch_state:log(iter)
      end
      if opt.save_every > 0 and iter % opt.save_every == 0 then
        checkpoint:save_iteration(iter, epoch_state, batch_order)
      end
      iter = iter + 1
    end

    return epoch_state
  end

  for epoch = opt.start_epoch, opt.epochs do
    local epoch_state = train_epoch(epoch)

    local valid_ppl = eval(model, criterion, valid_data)
    print('Validation PPL: ' .. valid_ppl)

    if opt.optim == 'sgd' then
      optim:update_learning_rate(valid_ppl, epoch)
    end

    log:append({epoch, epoch_state:get_time(), epoch_state:get_train_ppl(), valid_ppl, epoch_state:get_min_freememory()})

    checkpoint:save_epoch(valid_ppl, epoch_state, optim)
  end
end


local function main()
  local required_options = {
    "data",
    "save_file"
  }

  utils.Opt.init(opt, required_options)
  utils.Cuda.init(opt)
  utils.Parallel.init(opt)

  local log = utils.Log.new(opt.save_file .. ".log", not opt.no_log)

  -- Create the data loader class.
  print('Loading data from ' .. opt.data .. '...')
  local dataset = torch.load(opt.data)

  local train_data = Data.new(dataset.train.src, dataset.train.targ)
  local valid_data = Data.new(dataset.valid.src, dataset.valid.targ)

  train_data:set_batch_size(opt.max_batch_size)
  valid_data:set_batch_size(opt.max_batch_size)

  print(string.format(' * vocabulary size: source = %d; target = %d',
                      #dataset.dicts.src.words, #dataset.dicts.targ.words))
  print(string.format(' * additional features: source = %d; target = %d',
                      #dataset.dicts.src.features, #dataset.dicts.targ.features))
  print(string.format(' * maximum sequence length: source = %d; target = %d',
                      train_data.max_source_length, train_data.max_target_length))
  print(string.format(' * number of training sentences: %d', #train_data.src))
  print(string.format(' * maximum batch size: %d', opt.max_batch_size * utils.Parallel.count))

  local checkpoint = {}

  if opt.train_from:len() > 0 then
    assert(path.exists(opt.train_from), 'checkpoint path invalid')

    print('Loading checkpoint ' .. opt.train_from .. '...')
    checkpoint = torch.load(opt.train_from)

    opt.num_layers = checkpoint.options.num_layers
    opt.rnn_size = checkpoint.options.rnn_size
    opt.brnn = checkpoint.options.brnn
    opt.brnn_merge = checkpoint.options.brnn_merge
    opt.input_feed = checkpoint.options.input_feed

    -- Resume training from checkpoint
    if opt.train_from:len() > 0 and opt.continue then
      opt.optim = checkpoint.options.optim
      opt.lr_decay = checkpoint.options.lr_decay
      opt.start_decay_at = checkpoint.options.start_decay_at
      opt.epochs = checkpoint.options.epochs
      opt.curriculum = checkpoint.options.curriculum

      opt.learning_rate = checkpoint.info.learning_rate
      opt.optim_states = checkpoint.info.optim_states
      opt.start_epoch = checkpoint.info.epoch
      opt.start_iteration = checkpoint.info.iteration

      print('Resuming training from epoch ' .. opt.start_epoch
              .. ' at iteration ' .. opt.start_iteration .. '...')
    end
    log:append({'--- restart from checkpoint: ',opt.train_from})
  else
    log:clear()
  end

  print('Building model...')

  local model

  utils.Parallel.launch(nil, function(idx)

    _G.model = {}

    if checkpoint.models then
      _G.model.encoder = Models.loadEncoder(checkpoint.models.encoder, idx > 1)
      _G.model.decoder = Models.loadDecoder(checkpoint.models.decoder, idx > 1)
    else
      _G.model.encoder = Models.buildEncoder(opt, dataset.dicts.src)
      _G.model.decoder = Models.buildDecoder(opt, dataset.dicts.targ)
    end

    for _, mod in pairs(_G.model) do
      utils.Cuda.convert(mod)
    end

    return idx, _G.model
  end, function(idx, themodel)
    if idx == 1 then
      model = themodel
    end
  end)

  train_model(model, train_data, valid_data, dataset, checkpoint.info, log)
end

main()
