require 's2sa.utils.dict'

local path = require 'pl.path'
local cuda = require 's2sa.utils.cuda'

local Decoder = require 's2sa.decoder'
local Encoder = require 's2sa.encoder'
local BiEncoder = require 's2sa.biencoder'
local Generator = require 's2sa.generator'

local EpochState = require 's2sa.train.epoch_state'
local Checkpoint = require 's2sa.train.checkpoint'
local Data = require 's2sa.train.data'
local Optim = require 's2sa.train.optim'

local cmd = torch.CmdLine()

cmd:text("")
cmd:text("**Data options**")
cmd:text("")
cmd:option('-data','data/demo.t7', [[Path to the training *-train.t7 file from preprocess.lua]])
cmd:option('-savefile', 'seq2seq_lstm_attn', [[Savefile name (model will be saved as
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
cmd:option('-input_feed', true, [[Feed the context vector at each time step as additional input (via concatenation with the word embeddings) to the decoder.]])
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
cmd:option('-fallback_to_cpu', false, [[Fallback to CPU if no GPU available or can not use cuda/cudnn]])
cmd:option('-cudnn', false, [[Whether to use cudnn or not]])

-- bookkeeping
cmd:option('-save_every', 1, [[Save every this many epochs]])
cmd:option('-intermediate_save', 0, [[Save intermediate models every this many iterations within an epoch]])
cmd:option('-print_every', 50, [[Print stats after this many batches]])
cmd:option('-seed', 3435, [[Seed for random initialization]])

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
  nets.generator = model.generator.network

  return nets
end

local function init_params(nets)
  local num_params = 0
  local params = {}
  local grad_params = {}

  print('Initializing parameters...')

  for _, net in pairs(nets) do
    local p, gp = net:getParameters()

    if opt.train_from:len() == 0 then
      p:uniform(-opt.param_init, opt.param_init)
    end

    num_params = num_params + p:size(1)
    table.insert(params, p)
    table.insert(grad_params, gp:zero())
  end

  print(" * number of parameters: " .. num_params)

  return params, grad_params
end

local function eval(model, data)
  local loss = 0
  local total = 0

  model.encoder:evaluate()
  model.decoder:evaluate()
  model.generator:evaluate()

  for i = 1, #data do
    local batch = data:get_batch(i)

    local encoder_states, context = model.encoder:forward(batch)
    local decoder_outputs = model.decoder:forward(batch, encoder_states, context)

    loss = loss + model.generator:compute_loss(batch, decoder_outputs)
    total = total + batch.target_non_zeros
  end

  model.encoder:training()
  model.decoder:training()
  model.generator:training()

  return math.exp(loss / total)
end

local function train(model, train_data, valid_data, info)
  local nets = get_nets(model)
  local params, grad_params = init_params(nets)

  local optim = Optim.new({
    method = opt.optim,
    num_models = #params,
    learning_rate = opt.learning_rate,
    lr_decay = opt.lr_decay,
    start_decay_at = opt.start_decay_at,
    optim_states = opt.optim_states
  })

  local checkpoint = Checkpoint.new({
    options = opt,
    nets = nets,
    optim = optim
  })

  local function train_epoch(epoch)
    local epoch_state
    local batch_order

    local start_i = opt.start_iteration

    if start_i > 1 and info ~= nil then
      epoch_state = EpochState.new(epoch, info.epoch_status)
      batch_order = info.batch_order
    else
      epoch_state = EpochState.new(epoch)
      batch_order = torch.randperm(#train_data) -- shuffle mini batch order
    end

    opt.start_iteration = 1

    for i = start_i, #train_data do
      local batch_idx = batch_order[i]
      if epoch <= opt.curriculum then
        batch_idx = i
      end

      local batch = train_data:get_batch(batch_idx)

      local enc_states, context = model.encoder:forward(batch)
      local dec_outputs = model.decoder:forward(batch, enc_states, context)

      local enc_grad_states_out, grad_context, loss = model.decoder:backward(batch, dec_outputs, model.generator)
      model.encoder:backward(batch, enc_grad_states_out, grad_context)

      optim:update_params(params, grad_params, opt.max_grad_norm)
      epoch_state:update(batch, loss)

      if i % opt.print_every == 0 then
        epoch_state:log(i, #train_data, optim:get_learning_rate())
      end

      if opt.intermediate_save > 0 and i % opt.intermediate_save == 0 then
        checkpoint:save_iteration(i, epoch_state, batch_order)
      end
    end

    return epoch_state
  end

  for epoch = opt.start_epoch, opt.epochs do
    local epoch_state = train_epoch(epoch)

    local valid_ppl = eval(model, valid_data)
    print('Validation PPL: ' .. valid_ppl)

    if opt.optim == 'sgd' then
      optim:update_learning_rate(valid_ppl, epoch)
    end

    if epoch % opt.save_every == 0 then
      checkpoint:save_epoch(valid_ppl, epoch_state, optim)
    end
  end
end


local function main()
  torch.manualSeed(opt.seed)

  cuda.init(opt)

  -- Create the data loader class.
  print('Loading data from ' .. opt.data .. '...')
  local dataset = torch.load(opt.data)

  local train_data = Data.new(dataset.train, opt.max_batch_size)
  local valid_data = Data.new(dataset.valid, opt.max_batch_size)

  print(string.format(' * vocabluary size: source = %d; target = %d',
                      #dataset.src_dict, #dataset.targ_dict))
  print(string.format(' * maximum sequence length: source = %d; target = %d',
                      train_data.max_source_length, train_data.max_target_length))
  print(string.format(' * number of training sentences: %d', #train_data.src))
  print(string.format(' * number of batches: %d', #train_data))

  local checkpoint = {}
  checkpoint.nets = {}

  if opt.train_from:len() > 0 then
    assert(path.exists(opt.train_from), 'checkpoint path invalid')

    print('Loading checkpoint ' .. opt.train_from .. '...')
    checkpoint = torch.load(opt.train_from)

    opt.num_layers = checkpoint.options.num_layers
    opt.rnn_size = checkpoint.options.rnn_size
    opt.brnn = checkpoint.options.brnn
    opt.brnn_merge = checkpoint.options.brnn_merge
    opt.input_feed = checkpoint.options.input_feed

    -- resume training from checkpoint
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

      print('Resuming trainging from epoch ' .. opt.start_epoch
              .. ' at iteration ' .. opt.start_iteration .. '...')
    end
  end

  local encoder_args = {
    max_sent_length = math.max(train_data.max_source_length, valid_data.max_source_length),
    max_batch_size = opt.max_batch_size,
    word_vec_size = opt.word_vec_size,
    pre_word_vecs = opt.pre_word_vecs_enc,
    fix_word_vecs = opt.fix_word_vecs_enc,
    vocab_size = #dataset.src_dict,
    rnn_size = opt.rnn_size,
    dropout = opt.dropout,
    num_layers = opt.num_layers
  }

  local decoder_args = {
    max_sent_length = math.max(train_data.max_target_length, valid_data.max_target_length),
    max_source_length = math.max(train_data.max_source_length, valid_data.max_source_length),
    max_batch_size = opt.max_batch_size,
    word_vec_size = opt.word_vec_size,
    pre_word_vecs = opt.pre_word_vecs_dec,
    fix_word_vecs = opt.fix_word_vecs_dec,
    vocab_size = #dataset.targ_dict,
    rnn_size = opt.rnn_size,
    dropout = opt.dropout,
    num_layers = opt.num_layers,
    input_feed = opt.input_feed
  }

  local generator_args = {
    vocab_size = #dataset.targ_dict,
    rnn_size = opt.rnn_size
  }

  print('Building model...')
  local model = {}

  if opt.brnn then
    model.encoder = BiEncoder.new(encoder_args, opt.brnn_merge, checkpoint.nets.encoder, checkpoint.nets.encoder_bwd)
  else
    model.encoder = Encoder.new(encoder_args, checkpoint.nets.encoder)
  end

  model.decoder = Decoder.new(decoder_args, checkpoint.nets.decoder)
  model.generator = Generator.new(generator_args, checkpoint.nets.generator)

  for _, mod in pairs(model) do
    cuda.convert(mod)
    mod:training()
  end

  train(model, train_data, valid_data, checkpoint.info)
end

main()
