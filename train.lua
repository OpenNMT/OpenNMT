require 's2sa.dict'

local path = require 'pl.path'
local cuda = require 's2sa.cuda'
local Bookkeeper = require 's2sa.bookkeeper'
local Data = require 's2sa.data'
local Decoder = require 's2sa.decoder'
local Encoder = require 's2sa.encoder'
local Evaluator = require 's2sa.evaluator'
local Generator = require 's2sa.generator'
local Optim = require 's2sa.optim'

local cmd = torch.CmdLine()

cmd:text("")
cmd:text("**Data options**")
cmd:text("")
cmd:option('-data','data/demo.t7', [[Path to the training *.hdf5 file from preprocess.py]])
cmd:option('-savefile', 'seq2seq_lstm_attn', [[Savefile name (model will be saved as
                                             savefile_epochX_PPL.t7 where X is the X-th epoch and PPL is
                                             the validation perplexity]])
cmd:option('-train_from', '', [[If training from a checkpoint then this is the path to the pretrained model.]])

cmd:text("")
cmd:text("**Model options**")
cmd:text("")

cmd:option('-num_layers', 2, [[Number of layers in the LSTM encoder/decoder]])
cmd:option('-rnn_size', 500, [[Size of LSTM hidden states]])
cmd:option('-word_vec_size', 500, [[Word embedding sizes]])

cmd:text("")
cmd:text("**Optimization options**")
cmd:text("")

cmd:option('-max_batch_size', 64, [[Maximum batch size]])
cmd:option('-epochs', 13, [[Number of training epochs]])
cmd:option('-start_epoch', 1, [[If loading from a checkpoint, the epoch from which to start]])
cmd:option('-param_init', 0.1, [[Parameters are initialized over uniform distribution with support (-param_init, param_init)]])
cmd:option('-learning_rate', 1, [[Starting learning rate. If adagrad/adadelta/adam is used,
                                then this is the global learning rate. Recommended settings: sgd =1,
                                adagrad = 0.1, adadelta = 1, adam = 0.1]])
cmd:option('-max_grad_norm', 5, [[If the norm of the gradient vector exceeds this renormalize it to have the norm equal to max_grad_norm]])
cmd:option('-dropout', 0.3, [[Dropout probability. Dropout is applied between vertical LSTM stacks.]])
cmd:option('-lr_decay', 0.5, [[Decay learning rate by this much if (i) perplexity does not decrease
                             on the validation set or (ii) epoch has gone past the start_decay_at_limit]])
cmd:option('-start_decay_at', 9, [[Start decay after this epoch]])
cmd:option('-pre_word_vecs_enc', '', [[If a valid path is specified, then this will load
                                     pretrained word embeddings (hdf5 file) on the encoder side.
                                     See README for specific formatting instructions.]])
cmd:option('-pre_word_vecs_dec', '', [[If a valid path is specified, then this will load
                                     pretrained word embeddings (hdf5 file) on the decoder side.
                                     See README for specific formatting instructions.]])
cmd:option('-fix_word_vecs_enc', 0, [[If = 1, fix word embeddings on the encoder side]])
cmd:option('-fix_word_vecs_dec', 0, [[If = 1, fix word embeddings on the decoder side]])

cmd:text("")
cmd:text("**Other options**")
cmd:text("")

-- GPU
cmd:option('-gpuid', -1, [[Which gpu to use. < 1 = use CPU]])
cmd:option('-fallback_to_cpu', false, [[Fallback to CPU if no GPU available]])
cmd:option('-cudnn', false, [[Whether to use cudnn or not]])

-- bookkeeping
cmd:option('-save_every', 1, [[Save every this many epochs]])
cmd:option('-print_every', 50, [[Print stats after this many batches]])
cmd:option('-seed', 3435, [[Seed for random initialization]])

local opt = cmd:parse(arg)

local function save_model(model_path, data, options, double)
  print('saving model to ' .. model_path)
  if double then
    for i = 1, #data do data[i] = data[i]:double() end
  end
  torch.save(model_path, {data, options})
end

local function train(train_data, valid_data, encoder, decoder, generator)
  local num_params = 0
  local params = {}
  local grad_params = {}

  local layers = {encoder.network, decoder.network, generator.network}

  print('Initializing parameters...')
  for i = 1, #layers do
    local p, gp = layers[i]:getParameters()
    if opt.train_from:len() == 0 then
      p:uniform(-opt.param_init, opt.param_init)
    end
    num_params = num_params + p:size(1)
    params[i] = p
    grad_params[i] = gp
  end

  print("Number of parameters: " .. num_params)

  local function train_batch(data, epoch, optim)
    local bookkeeper = Bookkeeper.new({
      learning_rate = optim:get_rate(),
      data_size = #data,
      epoch = epoch
    })

    local batch_order = torch.randperm(#data) -- shuffle mini batch order

    for i = 1, #data do
      encoder:forget()
      decoder:forget()

      local batch = data:get_batch(batch_order[i])

      -- forward encoder
      local encoder_states, context = encoder:forward(batch)

      -- forward decoder
      local decoder_states, decoder_out = decoder:forward(batch, encoder_states)

      -- forward and backward attention and generator
      local decoder_grad_output, grad_context, loss = generator:process(batch, context, decoder_states, decoder_out)

      -- backward decoder
      local decoder_grad_input = decoder:backward(decoder_grad_output)

      -- backward encoder
      local encoder_grad_output = decoder_grad_input
      encoder_grad_output[#encoder_grad_output] = grad_context
      encoder:backward(encoder_grad_output)

      optim:update_params(params, grad_params, opt.max_grad_norm)

      -- Bookkeeping
      bookkeeper:update(batch, loss)

      if i % opt.print_every == 0 then
        bookkeeper:log(i)
      end
    end

    return bookkeeper:get_train_score()
  end

  local evaluator = Evaluator.new(opt.num_layers)
  local optim = Optim.new(opt.learning_rate, opt.lr_decay, opt.start_decay_at)

  opt.train_perf = {}

  for epoch = opt.start_epoch, opt.epochs do
    encoder:training()
    decoder:training()
    generator:training()

    local train_score = train_batch(train_data, epoch, optim)

    print('Train', train_score)
    opt.train_perf[#opt.train_perf + 1] = train_score

    local score = evaluator:process({
      encoder = encoder,
      decoder = decoder,
      generator = generator
    }, valid_data)

    optim:update_rate(score, epoch)

    -- clean and save models
    if epoch % opt.save_every == 0 then
      save_model(string.format('%s_epoch%.2f_%.2f.t7', opt.savefile, epoch, score), {encoder, decoder, generator}, opt, false)
    end
  end
  -- save final model
  save_model(string.format('%s_final.t7', opt.savefile), {encoder, decoder, generator}, opt, true)
end

local function main()
  torch.manualSeed(opt.seed)

  cuda.init(opt)

  -- Create the data loader class.
  print('Loading data from ' .. opt.data .. '...')
  local dataset = torch.load(opt.data)

  local train_data = Data.new(dataset.train, opt.max_batch_size)
  local valid_data = Data.new(dataset.valid, opt.max_batch_size)

  print(string.format('Source vocab size: %d, Target vocab size: %d',
                      #dataset.src_dict, #dataset.targ_dict))
  print(string.format('Source max sent len: %d, Target max sent len: %d',
                      train_data.max_source_length, train_data.max_target_length))

  -- Build model
  local encoder
  local decoder
  local generator

  if opt.train_from:len() == 0 then
    encoder = Encoder.new({
      pre_word_vecs = opt.pre_word_vecs_enc,
      fix_word_vecs = opt.fix_word_vecs_enc,
      vocab_size = #dataset.src_dict
    }, opt)

    decoder = Decoder.new({
      pre_word_vecs = opt.pre_word_vecs_dec,
      fix_word_vecs = opt.fix_word_vec,
      vocab_size = #dataset.targ_dict
    }, opt)

    generator = Generator.new({
      vocab_size = #dataset.targ_dict
    }, opt)
  else
    assert(path.exists(opt.train_from), 'checkpoint path invalid')
    print('loading ' .. opt.train_from .. '...')
    local checkpoint = torch.load(opt.train_from)
    local model, model_opt = checkpoint[1], checkpoint[2]
    opt.num_layers = model_opt.num_layers
    opt.rnn_size = model_opt.rnn_size
    encoder = model[1]
    decoder = model[2]
    generator = model[3]
  end

  generator:build_criterion(#dataset.targ_dict)

  cuda.convert({encoder.network, decoder.network, generator.network, generator.criterion})

  train(train_data, valid_data, encoder, decoder, generator)
end

main()
