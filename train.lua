require 's2sa.dict'
local hdf5 = require 'hdf5'
local path = require 'pl.path'

local models = require 's2sa.models'
local model_utils = require 's2sa.model_utils'
local table_utils = require 's2sa.table_utils'

local Bookkeeper = require 's2sa.bookkeeper'
local Data = require 's2sa.data'
local Evaluator = require 's2sa.evaluator'
local Learning = require 's2sa.learning'

local cmd = torch.CmdLine()
local opt = {}
local layers = {}
local word_vecs_enc = {}
local word_vecs_dec = {}
local encoder
local decoder
local generator
local criterion

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

-- bookkeeping
cmd:option('-save_every', 1, [[Save every this many epochs]])
cmd:option('-print_every', 50, [[Print stats after this many batches]])
cmd:option('-seed', 3435, [[Seed for random initialization]])


local function save_model(model_path, data, options, double)
  print('saving model to ' .. model_path)
  if double then
    for i = 1, #data do data[i] = data[i]:double() end
  end
  torch.save(model_path, {data, options})
end

local function train(train_data, valid_data)
  local num_params = 0
  local num_prunedparams = 0
  local params, grad_params = {}, {}
  opt.train_perf = {}

  for i = 1, #layers do
    local p, gp = layers[i]:getParameters()
    if opt.train_from:len() == 0 then
      p:uniform(-opt.param_init, opt.param_init)
    end
    num_params = num_params + p:size(1)
    params[i] = p
    grad_params[i] = gp
  end

  if opt.pre_word_vecs_enc:len() > 0 then
    local f = hdf5.open(opt.pre_word_vecs_enc)
    local pre_word_vecs = f:read('word_vecs'):all()
    for i = 1, pre_word_vecs:size(1) do
      word_vecs_enc.weight[i]:copy(pre_word_vecs[i])
    end
  end
  if opt.pre_word_vecs_dec:len() > 0 then
    local f = hdf5.open(opt.pre_word_vecs_dec)
    local pre_word_vecs = f:read('word_vecs'):all()
    for i = 1, pre_word_vecs:size(1) do
      word_vecs_dec.weight[i]:copy(pre_word_vecs[i])
    end
  end

  print("Number of parameters: " .. num_params .. " (active: " .. num_params-num_prunedparams .. ")")

  word_vecs_enc.weight[1]:zero()
  word_vecs_dec.weight[1]:zero()

  local max_length = math.max(opt.max_source_length, opt.max_target_length)

  -- prototypes for gradients so there is no need to clone
  local encoder_grad_proto = torch.zeros(opt.max_batch_size, max_length, opt.rnn_size)
  local context_proto = torch.zeros(opt.max_batch_size, max_length, opt.rnn_size)

  -- clone encoder/decoder up to max source/target length
  local decoder_clones = model_utils.clone_many_times(decoder, opt.max_target_length)
  local encoder_clones = model_utils.clone_many_times(encoder, opt.max_source_length)

  local h_init = torch.zeros(opt.max_batch_size, opt.rnn_size)
  if opt.gpuid > 0 then
    h_init = h_init:cuda()
    cutorch.setDevice(opt.gpuid)
    context_proto = context_proto:cuda()
    encoder_grad_proto = encoder_grad_proto:cuda()
  end

  -- these are initial states of encoder/decoder for fwd/bwd steps
  local init_fwd_enc = {}
  local init_bwd_enc = {}
  local init_fwd_dec = {}
  local init_bwd_dec = {}

  for _ = 1, opt.num_layers do
    table.insert(init_fwd_enc, h_init:clone())
    table.insert(init_fwd_enc, h_init:clone())
    table.insert(init_bwd_enc, h_init:clone())
    table.insert(init_bwd_enc, h_init:clone())
  end
  table.insert(init_bwd_dec, h_init:clone())
  for _ = 1, opt.num_layers do
    table.insert(init_fwd_dec, h_init:clone())
    table.insert(init_fwd_dec, h_init:clone())
    table.insert(init_bwd_dec, h_init:clone())
    table.insert(init_bwd_dec, h_init:clone())
  end

  local dec_offset = 3 -- offset depends on input feeding


  function train_batch(data, epoch, learning)
    local bookkeeper = Bookkeeper.new({
      print_frequency = opt.print_every,
      learning_rate = learning:get_rate(),
      data_size = #data,
      epoch = epoch
    })

    local batch_order = torch.randperm(#data) -- shuffle mini batch order

    for i = 1, #data do
      table_utils.zero(grad_params, 'zero')

      local batch = data:get_batch(batch_order[i])

      local encoder_grads = encoder_grad_proto[{{1, batch.size}, {1, batch.source_length}}]
      local rnn_state_enc = model_utils.reset_state(init_fwd_enc, batch.size, 0)
      local context = context_proto[{{1, batch.size}, {1, batch.source_length}}]

      -- forward prop encoder
      for t = 1, batch.source_length do
        encoder_clones[t]:training()
        local encoder_input = {batch.source_input[t]}
        table_utils.append(encoder_input, rnn_state_enc[t-1])
        local out = encoder_clones[t]:forward(encoder_input)
        rnn_state_enc[t] = out
        context[{{},t}]:copy(out[#out])
      end

      -- copy encoder last hidden state to decoder initial state
      local rnn_state_dec = model_utils.reset_state(init_fwd_dec, batch.size, 0)
      for L = 1, opt.num_layers do
        rnn_state_dec[0][L*2-1]:copy(rnn_state_enc[batch.source_length][L*2-1])
        rnn_state_dec[0][L*2]:copy(rnn_state_enc[batch.source_length][L*2])
      end

      -- forward prop decoder
      local preds = {}

      for t = 1, batch.target_length do
        decoder_clones[t]:training()
        local decoder_input = {batch.target_input[t], context, table.unpack(rnn_state_dec[t-1])}
        local out = decoder_clones[t]:forward(decoder_input)
        local next_state = {}
        table.insert(preds, out[#out])
        for j = 1, #out-1 do
          table.insert(next_state, out[j])
        end
        rnn_state_dec[t] = next_state
      end

      -- backward prop decoder
      encoder_grads:zero()

      local drnn_state_dec = model_utils.reset_state(init_bwd_dec, batch.size)
      local loss = 0
      for t = batch.target_length, 1, -1 do
        local pred = generator:forward(preds[t])

        local input = pred
        local output = batch.target_output[t]

        loss = loss + criterion:forward(input, output)/batch.size

        local dl_dpred = criterion:backward(input, output)

        dl_dpred:div(batch.size)
        local dl_dtarget = generator:backward(preds[t], dl_dpred)

        local rnn_state_dec_pred_idx = #drnn_state_dec
        drnn_state_dec[rnn_state_dec_pred_idx]:add(dl_dtarget)

        local decoder_input = {batch.target_input[t], context, table.unpack(rnn_state_dec[t-1])}
        local dlst = decoder_clones[t]:backward(decoder_input, drnn_state_dec)
        -- accumulate encoder/decoder grads
        encoder_grads:add(dlst[2])

        drnn_state_dec[rnn_state_dec_pred_idx]:zero()
        for j = dec_offset, #dlst do
          drnn_state_dec[j-dec_offset+1]:copy(dlst[j])
        end
      end

      word_vecs_dec.gradWeight[1]:zero()
      if opt.fix_word_vecs_dec == 1 then
        word_vecs_dec.gradWeight:zero()
      end

      local grad_norm = 0
      grad_norm = grad_norm + grad_params[2]:norm()^2 + grad_params[3]:norm()^2

      local drnn_state_enc = model_utils.reset_state(init_bwd_enc, batch.size)
      for L = 1, opt.num_layers do
        drnn_state_enc[L*2-1]:copy(drnn_state_dec[L*2-1])
        drnn_state_enc[L*2]:copy(drnn_state_dec[L*2])
      end

      for t = batch.source_length, 1, -1 do
        local encoder_input = {batch.source_input[t]}
        table_utils.append(encoder_input, rnn_state_enc[t-1])
        drnn_state_enc[#drnn_state_enc]:add(encoder_grads[{{},t}])
        local dlst = encoder_clones[t]:backward(encoder_input, drnn_state_enc)
        for j = 1, #drnn_state_enc do
          drnn_state_enc[j]:copy(dlst[j+1])
        end
      end

      word_vecs_enc.gradWeight[1]:zero()
      if opt.fix_word_vecs_enc == 1 then
        word_vecs_enc.gradWeight:zero()
      end

      grad_norm = grad_norm + grad_params[1]:norm()^2
      if opt.brnn == 1 then
        grad_norm = grad_norm + grad_params[4]:norm()^2
      end
      grad_norm = grad_norm^0.5

      -- Shrink norm and update params
      local param_norm = 0
      local shrinkage = opt.max_grad_norm / grad_norm
      for j = 1, #grad_params do
        if shrinkage < 1 then
          grad_params[j]:mul(shrinkage)
        end
        params[j]:add(grad_params[j]:mul(-learning:get_rate()))
        param_norm = param_norm + params[j]:norm()^2
      end
      param_norm = param_norm^0.5

      -- Bookkeeping
      bookkeeper:update({
        source_size = batch.source_length,
        target_size = batch.target_length,
        batch_size = batch.size,
        batch_index = i,
        nonzeros = batch.target_non_zeros,
        loss = loss,
        param_norm = param_norm,
        grad_norm = grad_norm
      })

      if i % 200 == 0 then
        collectgarbage()
      end
    end

    return bookkeeper:get_train_score()
  end

  local evaluator = Evaluator.new(opt.num_layers)
  local learning = Learning.new(opt.learning_rate, opt.lr_decay, opt.start_decay_at)

  for epoch = opt.start_epoch, opt.epochs do
    generator:training()
    local train_score = train_batch(train_data, epoch, learning)

    print('Train', train_score)
    opt.train_perf[#opt.train_perf + 1] = train_score

    local score = evaluator:process({
      encoder = encoder_clones[1],
      decoder = decoder_clones[1],
      generator = generator,
      init_fwd_enc = init_fwd_enc,
      init_fwd_dec = init_fwd_dec,
      context_proto = context_proto,
      criterion = criterion
    }, valid_data)
    learning:update_rate(score, epoch)

    -- clean and save models
    if epoch % opt.save_every == 0 then
      save_model(string.format('%s_epoch%.2f_%.2f.t7', opt.savefile, epoch, score), {encoder, decoder, generator}, opt, false)
    end
  end
  -- save final model
  save_model(string.format('%s_final.t7', opt.savefile), {encoder, decoder, generator}, opt, true)
end

local function get_layer(layer)
  if layer.name ~= nil then
    if layer.name == 'word_vecs_dec' then
      word_vecs_dec = layer
    elseif layer.name == 'word_vecs_enc' then
       word_vecs_enc = layer
    end
  end
end

local function main()
  -- parse input params
  opt = cmd:parse(arg)

  torch.manualSeed(opt.seed)

  local cuda = opt.gpuid > 0
  if cuda then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpuid)
    cutorch.manualSeed(opt.seed)
  end

  -- Create the data loader class.
  print('Loading data from ' .. opt.data .. '...')
  local dataset = torch.load(opt.data)
  local train_data = Data.new(dataset.train, opt.max_batch_size, cuda)
  local valid_data = Data.new(dataset.valid, opt.max_batch_size, cuda)
  print('... done')

  opt.max_source_length = math.max(train_data.max_source_length, valid_data.max_source_length)
  opt.max_target_length = math.max(train_data.max_target_length, valid_data.max_target_length)

  print(string.format('Source vocab size: %d, Target vocab size: %d', #dataset.src_dict, #dataset.targ_dict))
  print(string.format('Source max sent len: %d, Target max sent len: %d',
                      opt.max_source_length, opt.max_target_length))

  -- Build model
  if opt.train_from:len() == 0 then
    encoder = models.make_lstm(#dataset.src_dict, opt, 'enc')
    decoder = models.make_lstm(#dataset.targ_dict, opt, 'dec')
    criterion, generator = models.make_generator(#dataset.targ_dict, opt)
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
    criterion = models.make_generator(valid_data, opt)
  end

  layers = {encoder, decoder, generator}

  if cuda then
    for i = 1, #layers do
      layers[i]:cuda()
    end
    criterion:cuda()
  end

  -- these layers will be manipulated during training
  encoder:apply(get_layer)
  decoder:apply(get_layer)
  train(train_data, valid_data)
end

main()
