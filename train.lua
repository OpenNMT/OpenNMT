require 'nn'
require 'nngraph'
require 'hdf5'

require 's2sa.data'
require 's2sa.models'
require 's2sa.model_utils'

cmd = torch.CmdLine()

-- data files
cmd:text("")
cmd:text("**Data options**")
cmd:text("")
cmd:option('-data_file','data/demo-train.hdf5', [[Path to the training *.hdf5 file from preprocess.py]])
cmd:option('-val_data_file','data/demo-val.hdf5', [[Path to validation *.hdf5 file from preprocess.py]])
cmd:option('-savefile', 'seq2seq_lstm_attn', [[Savefile name (model will be saved as
                                             savefile_epochX_PPL.t7 where X is the X-th epoch and PPL is
                                             the validation perplexity]])
cmd:option('-num_shards', 0, [[If the training data has been broken up into different shards,
                             then training files are in this many partitions]])
cmd:option('-train_from', '', [[If training from a checkpoint then this is the path to the pretrained model.]])

-- rnn model specs
cmd:text("")
cmd:text("**Model options**")
cmd:text("")

cmd:option('-num_layers', 2, [[Number of layers in the LSTM encoder/decoder]])
cmd:option('-rnn_size', 500, [[Size of LSTM hidden states]])
cmd:option('-word_vec_size', 500, [[Word embedding sizes]])
cmd:option('-attn', 1, [[If = 1, use attention on the decoder side. If = 0, it uses the last
                       hidden state of the decoder as context at each time step.]])
cmd:option('-brnn', 0, [[If = 1, use a bidirectional RNN. Hidden states of the fwd/bwd RNNs are summed.]])
cmd:option('-use_chars_enc', 0, [[If = 1, use character on the encoder side (instead of word embeddings]])
cmd:option('-use_chars_dec', 0, [[If = 1, use character on the decoder side (instead of word embeddings]])
cmd:option('-reverse_src', 0, [[If = 1, reverse the source sequence. The original
                              sequence-to-sequence paper found that this was crucial to
                              achieving good performance, but with attention models this
                              does not seem necessary. Recommend leaving it to 0]])
cmd:option('-init_dec', 1, [[Initialize the hidden/cell state of the decoder at time
                           0 to be the last hidden/cell state of the encoder. If 0,
                           the initial states of the decoder are set to zero vectors]])
cmd:option('-input_feed', 1, [[If = 1, feed the context vector at each time step as additional
                             input (vica concatenation with the word embeddings) to the decoder]])
cmd:option('-multi_attn', 0, [[If > 0, then use a another attention layer on this layer of
                             the decoder. For example, if num_layers = 3 and `multi_attn = 2`,
                             then the model will do an attention over the source sequence
                             on the second layer (and use that as input to the third layer) and
                             the penultimate layer]])
cmd:option('-res_net', 0, [[Use residual connections between LSTM stacks whereby the input to
                          the l-th LSTM layer if the hidden state of the l-1-th LSTM layer
                          added with the l-2th LSTM layer. We didn't find this to help in our
                          experiments]])
cmd:option('-guided_alignment', 0, [[If 1, use external alignments to guide the attention weights as in
                                   (Chen et al., Guided Alignment Training for Topic-Aware Neural Machine Translation,
                                   arXiv 2016.). Alignments should have been provided during preprocess]])
cmd:option('-guided_alignment_weight', 0.5, [[default weights for external alignments]])
cmd:option('-guided_alignment_decay', 1, [[decay rate per epoch for alignment weight - typical with 0.9,
                                         weight will end up at ~30% of its initial value]])

cmd:text("")
cmd:text("Below options only apply if using the character model.")
cmd:text("")

-- char-cnn model specs (if use_chars == 1)
cmd:option('-char_vec_size', 25, [[Size of the character embeddings]])
cmd:option('-kernel_width', 6, [[Size (i.e. width) of the convolutional filter]])
cmd:option('-num_kernels', 1000, [[Number of convolutional filters (feature maps). So the
                                 representation from characters will have this many dimensions]])
cmd:option('-num_highway_layers', 2, [[Number of highway layers in the character model]])

cmd:text("")
cmd:text("**Optimization options**")
cmd:text("")

-- optimization
cmd:option('-epochs', 13, [[Number of training epochs]])
cmd:option('-start_epoch', 1, [[If loading from a checkpoint, the epoch from which to start]])
cmd:option('-param_init', 0.1, [[Parameters are initialized over uniform distribution with support (-param_init, param_init)]])
cmd:option('-optim', 'sgd', [[Optimization method. Possible options are: sgd (vanilla SGD), adagrad, adadelta, adam]])
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
cmd:option('-feature_embeddings_dim_exponent', 0.7, [[If the feature takes N values, then the
                                                    embbeding dimension will be set to N^exponent]])
cmd:option('-pre_word_vecs_enc', '', [[If a valid path is specified, then this will load
                                     pretrained word embeddings (hdf5 file) on the encoder side.
                                     See README for specific formatting instructions.]])
cmd:option('-pre_word_vecs_dec', '', [[If a valid path is specified, then this will load
                                     pretrained word embeddings (hdf5 file) on the decoder side.
                                     See README for specific formatting instructions.]])
cmd:option('-fix_word_vecs_enc', 0, [[If = 1, fix word embeddings on the encoder side]])
cmd:option('-fix_word_vecs_dec', 0, [[If = 1, fix word embeddings on the decoder side]])
cmd:option('-max_batch_l', '', [[If blank, then it will infer the max batch size from validation
                               data. You should only use this if your validation set uses a different
                               batch size in the preprocessing step]])

cmd:text("")
cmd:text("**Other options**")
cmd:text("")

cmd:option('-start_symbol', 0, [[Use special start-of-sentence and end-of-sentence tokens
                               on the source side. We've found this to make minimal difference]])
-- GPU
cmd:option('-gpuid', -1, [[Which gpu to use. -1 = use CPU]])
cmd:option('-gpuid2', -1, [[If this is >= 0, then the model will use two GPUs whereby the encoder
                          is on the first GPU and the decoder is on the second GPU.
                          This will allow you to train with bigger batches/models.]])
cmd:option('-cudnn', 0, [[Whether to use cudnn or not for convolutions (for the character model).
                        cudnn has much faster convolutions so this is highly recommended
                        if using the character model]])
-- bookkeeping
cmd:option('-save_every', 1, [[Save every this many epochs]])
cmd:option('-print_every', 50, [[Print stats after this many batches]])
cmd:option('-seed', 3435, [[Seed for random initialization]])
cmd:option('-prealloc', 1, [[Use memory preallocation and sharing between cloned encoder/decoders]])

function zero_table(t)
  for i = 1, #t do
    if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
      if i == 1 then
        cutorch.setDevice(opt.gpuid)
      else
        cutorch.setDevice(opt.gpuid2)
      end
    end
    t[i]:zero()
  end
end

function append_table(dst, src)
  for i = 1, #src do
    table.insert(dst, src[i])
  end
end

function train(train_data, valid_data)

  local timer = torch.Timer()
  local num_params = 0
  local num_prunedparams = 0
  local start_decay = 0
  params, grad_params = {}, {}
  opt.train_perf = {}
  opt.val_perf = {}

  for i = 1, #layers do
    if opt.gpuid2 >= 0 then
      if i == 1 then
        cutorch.setDevice(opt.gpuid)
      else
        cutorch.setDevice(opt.gpuid2)
      end
    end
    local p, gp = layers[i]:getParameters()
    if opt.train_from:len() == 0 then
      p:uniform(-opt.param_init, opt.param_init)
    end
    num_params = num_params + p:size(1)
    params[i] = p
    grad_params[i] = gp
    layers[i]:apply(function (m) if m.nPruned then num_prunedparams=num_prunedparams+m:nPruned() end end)
  end

  if opt.pre_word_vecs_enc:len() > 0 then
    local f = hdf5.open(opt.pre_word_vecs_enc)
    local pre_word_vecs = f:read('word_vecs'):all()
    for i = 1, pre_word_vecs:size(1) do
      word_vec_layers[1].weight[i]:copy(pre_word_vecs[i])
    end
  end
  if opt.pre_word_vecs_dec:len() > 0 then
    local f = hdf5.open(opt.pre_word_vecs_dec)
    local pre_word_vecs = f:read('word_vecs'):all()
    for i = 1, pre_word_vecs:size(1) do
      word_vec_layers[2].weight[i]:copy(pre_word_vecs[i])
    end
  end
  if opt.brnn == 1 then --subtract shared params for brnn
    num_params = num_params - word_vec_layers[1].weight:nElement()
    word_vec_layers[3].weight:copy(word_vec_layers[1].weight)
    if opt.use_chars_enc == 1 then
      for i = 1, charcnn_offset do
        num_params = num_params - charcnn_layers[i]:nElement()
        charcnn_layers[i+charcnn_offset]:copy(charcnn_layers[i])
      end
    end
  end

  print("Number of parameters: " .. num_params .. " (active: " .. num_params-num_prunedparams .. ")")

  if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
    cutorch.setDevice(opt.gpuid)
    word_vec_layers[1].weight[1]:zero()
    cutorch.setDevice(opt.gpuid2)
    word_vec_layers[2].weight[1]:zero()
  else
    word_vec_layers[1].weight[1]:zero()
    word_vec_layers[2].weight[1]:zero()
    if opt.brnn == 1 then
      word_vec_layers[3].weight[1]:zero()
    end
  end

  -- prototypes for gradients so there is no need to clone
  encoder_grad_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.rnn_size)
  encoder_bwd_grad_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.rnn_size)
  context_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.rnn_size)
  -- need more copies of the above if using two gpus
  if opt.gpuid2 >= 0 then
    encoder_grad_proto2 = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.rnn_size)
    context_proto2 = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.rnn_size)
    encoder_bwd_grad_proto2 = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.rnn_size)
  end

  -- clone encoder/decoder up to max source/target length
  decoder_clones = clone_many_times(decoder, opt.max_sent_l_targ)
  encoder_clones = clone_many_times(encoder, opt.max_sent_l_src)
  if opt.brnn == 1 then
    encoder_bwd_clones = clone_many_times(encoder_bwd, opt.max_sent_l_src)
  end
  for i = 1, opt.max_sent_l_src do
    if encoder_clones[i].apply then
      encoder_clones[i]:apply(function(m) m:setReuse() end)
      if opt.prealloc == 1 then encoder_clones[i]:apply(function(m) m:setPrealloc() end) end
    end
    if opt.brnn == 1 then
      encoder_bwd_clones[i]:apply(function(m) m:setReuse() end)
      if opt.prealloc == 1 then encoder_bwd_clones[i]:apply(function(m) m:setPrealloc() end) end
    end
  end
  for i = 1, opt.max_sent_l_targ do
    if decoder_clones[i].apply then
      decoder_clones[i]:apply(function(m) m:setReuse() end)
      if opt.prealloc == 1 then decoder_clones[i]:apply(function(m) m:setPrealloc() end) end
    end
  end

  local h_init = torch.zeros(opt.max_batch_l, opt.rnn_size)
  local attn_init = torch.zeros(opt.max_batch_l, opt.max_sent_l)
  if opt.gpuid >= 0 then
    h_init = h_init:cuda()
    attn_init = attn_init:cuda()
    cutorch.setDevice(opt.gpuid)
    if opt.gpuid2 >= 0 then
      encoder_grad_proto2 = encoder_grad_proto2:cuda()
      encoder_bwd_grad_proto2 = encoder_bwd_grad_proto2:cuda()
      context_proto = context_proto:cuda()
      cutorch.setDevice(opt.gpuid2)
      encoder_grad_proto = encoder_grad_proto:cuda()
      encoder_bwd_grad_proto = encoder_bwd_grad_proto:cuda()
      context_proto2 = context_proto2:cuda()
      cutorch.setDevice(opt.gpuid)
    else
      context_proto = context_proto:cuda()
      encoder_grad_proto = encoder_grad_proto:cuda()
      if opt.brnn == 1 then
        encoder_bwd_grad_proto = encoder_bwd_grad_proto:cuda()
      end
    end
  end

  -- these are initial states of encoder/decoder for fwd/bwd steps
  init_fwd_enc = {}
  init_bwd_enc = {}
  init_fwd_dec = {}
  init_bwd_dec = {}

  for L = 1, opt.num_layers do
    table.insert(init_fwd_enc, h_init:clone())
    table.insert(init_fwd_enc, h_init:clone())
    table.insert(init_bwd_enc, h_init:clone())
    table.insert(init_bwd_enc, h_init:clone())
  end
  if opt.gpuid2 >= 0 then
    cutorch.setDevice(opt.gpuid2)
  end
  if opt.input_feed == 1 then
    table.insert(init_fwd_dec, h_init:clone())
  end
  table.insert(init_bwd_dec, h_init:clone())
  for L = 1, opt.num_layers do
    table.insert(init_fwd_dec, h_init:clone())
    table.insert(init_fwd_dec, h_init:clone())
    table.insert(init_bwd_dec, h_init:clone())
    table.insert(init_bwd_dec, h_init:clone())
  end

  dec_offset = 3 -- offset depends on input feeding
  if opt.input_feed == 1 then
    dec_offset = dec_offset + 1
  end

  function reset_state(state, batch_l, t)
    if t == nil then
      local u = {}
      for i = 1, #state do
        state[i]:zero()
        table.insert(u, state[i][{{1, batch_l}}])
      end
      return u
    else
      local u = {[t] = {}}
      for i = 1, #state do
        state[i]:zero()
        table.insert(u[t], state[i][{{1, batch_l}}])
      end
      return u
    end
  end

  -- clean layer before saving to make the model smaller
  function clean_layer(layer)
    if opt.gpuid >= 0 then
      layer.output = torch.CudaTensor()
      layer.gradInput = torch.CudaTensor()
    else
      layer.output = torch.DoubleTensor()
      layer.gradInput = torch.DoubleTensor()
    end
    if layer.modules then
      for i, mod in ipairs(layer.modules) do
        clean_layer(mod)
      end
    elseif torch.type(self) == "nn.gModule" then
      layer:apply(clean_layer)
    end
  end

  -- decay learning rate if val perf does not improve or we hit the opt.start_decay_at limit
  function decay_lr(epoch)
    print(opt.val_perf)
    if epoch >= opt.start_decay_at then
      start_decay = 1
    end

    if opt.val_perf[#opt.val_perf] ~= nil and opt.val_perf[#opt.val_perf-1] ~= nil then
      local curr_ppl = opt.val_perf[#opt.val_perf]
      local prev_ppl = opt.val_perf[#opt.val_perf-1]
      if curr_ppl > prev_ppl then
        start_decay = 1
      end
    end
    if start_decay == 1 then
      opt.learning_rate = opt.learning_rate * opt.lr_decay
    end
  end

  function train_batch(data, epoch)
    opt.num_source_features = data.num_source_features

    local train_nonzeros = 0
    local train_loss = 0
    local train_loss_cll = 0
    local batch_order = torch.randperm(data.length) -- shuffle mini batch order
    local start_time = timer:time().real
    local num_words_target = 0
    local num_words_source = 0

    for i = 1, data:size() do
      zero_table(grad_params, 'zero')
      local d
      if epoch <= opt.curriculum then
        d = data[i]
      else
        d = data[batch_order[i]]
      end
      local target, target_out, nonzeros, source = d[1], d[2], d[3], d[4]
      local batch_l, target_l, source_l = d[5], d[6], d[7]
      local source_features = d[9]
      local alignment = d[10]
      local norm_alignment
      if opt.guided_alignment == 1 then
        replicator=nn.Replicate(alignment:size(2),2)
        if opt.gpuid >= 0 then
          cutorch.setDevice(opt.gpuid)
          if opt.gpuid2 >= 0 then -- alignment is in the 2nd GPU
            cutorch.setDevice(opt.gpuid2)
          end
          replicator = replicator:cuda()
        end
        norm_alignment = torch.cdiv(alignment, replicator:forward(torch.sum(alignment,2):squeeze(2)))
        norm_alignment[norm_alignment:ne(norm_alignment)] = 0
      end

      local encoder_grads = encoder_grad_proto[{{1, batch_l}, {1, source_l}}]
      local encoder_bwd_grads
      if opt.brnn == 1 then
        encoder_bwd_grads = encoder_bwd_grad_proto[{{1, batch_l}, {1, source_l}}]
      end
      if opt.gpuid >= 0 then
        cutorch.setDevice(opt.gpuid)
      end
      local rnn_state_enc = reset_state(init_fwd_enc, batch_l, 0)
      local context = context_proto[{{1, batch_l}, {1, source_l}}]
      -- forward prop encoder
      for t = 1, source_l do
        encoder_clones[t]:training()
        local encoder_input = {source[t]}
        if data.num_source_features > 0 then
          append_table(encoder_input, source_features[t])
        end
        append_table(encoder_input, rnn_state_enc[t-1])
        local out = encoder_clones[t]:forward(encoder_input)
        rnn_state_enc[t] = out
        context[{{},t}]:copy(out[#out])
      end

      local rnn_state_enc_bwd
      if opt.brnn == 1 then
        rnn_state_enc_bwd = reset_state(init_fwd_enc, batch_l, source_l+1)
        for t = source_l, 1, -1 do
          encoder_bwd_clones[t]:training()
          local encoder_input = {source[t]}
          if data.num_source_features > 0 then
            append_table(encoder_input, source_features[t])
          end
          append_table(encoder_input, rnn_state_enc_bwd[t+1])
          local out = encoder_bwd_clones[t]:forward(encoder_input)
          rnn_state_enc_bwd[t] = out
          context[{{},t}]:add(out[#out])
        end
      end

      if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
        cutorch.setDevice(opt.gpuid2)
        local context2 = context_proto2[{{1, batch_l}, {1, source_l}}]
        context2:copy(context)
        context = context2
      end
      -- copy encoder last hidden state to decoder initial state
      local rnn_state_dec = reset_state(init_fwd_dec, batch_l, 0)
      if opt.init_dec == 1 then
        for L = 1, opt.num_layers do
          rnn_state_dec[0][L*2-1+opt.input_feed]:copy(rnn_state_enc[source_l][L*2-1])
          rnn_state_dec[0][L*2+opt.input_feed]:copy(rnn_state_enc[source_l][L*2])
        end
        if opt.brnn == 1 then
          for L = 1, opt.num_layers do
            rnn_state_dec[0][L*2-1+opt.input_feed]:add(rnn_state_enc_bwd[1][L*2-1])
            rnn_state_dec[0][L*2+opt.input_feed]:add(rnn_state_enc_bwd[1][L*2])
          end
        end
      end
      -- forward prop decoder
      local preds = {}
      local attn_outputs = {}
      local decoder_input
      for t = 1, target_l do
        decoder_clones[t]:training()
        local decoder_input
        if opt.attn == 1 then
          decoder_input = {target[t], context, table.unpack(rnn_state_dec[t-1])}
        else
          decoder_input = {target[t], context[{{}, source_l}], table.unpack(rnn_state_dec[t-1])}
        end
        local out = decoder_clones[t]:forward(decoder_input)
        local out_pred_idx = #out
        if opt.guided_alignment == 1 then
          out_pred_idx = #out-1
          table.insert(attn_outputs, out[#out])
        end
        local next_state = {}
        table.insert(preds, out[out_pred_idx])
        if opt.input_feed == 1 then
          table.insert(next_state, out[out_pred_idx])
        end
        for j = 1, out_pred_idx-1 do
          table.insert(next_state, out[j])
        end
        rnn_state_dec[t] = next_state
      end

      -- backward prop decoder
      encoder_grads:zero()
      if opt.brnn == 1 then
        encoder_bwd_grads:zero()
      end

      local drnn_state_dec = reset_state(init_bwd_dec, batch_l)
      if opt.guided_alignment == 1 then
        attn_init:zero()
        table.insert(drnn_state_dec, attn_init[{{1, batch_l}, {1, source_l}}])
      end
      local loss = 0
      local loss_cll = 0
      for t = target_l, 1, -1 do
        local pred = generator:forward(preds[t])

        local input = pred
        local output = target_out[t]
        if opt.guided_alignment == 1 then
          input={input, attn_outputs[t]}
          output={output, norm_alignment[{{},{},t}]}
        end

        loss = loss + criterion:forward(input, output)/batch_l

        local drnn_state_attn
        local dl_dpred
        if opt.guided_alignment == 1 then
          local dl_dpred_attn = criterion:backward(input, output)
          dl_dpred = dl_dpred_attn[1]
          drnn_state_attn = dl_dpred_attn[2]
          drnn_state_attn:div(batch_l)
          loss_cll = loss_cll + cll_criterion:forward(input[1], output[1])/batch_l
        else
          dl_dpred = criterion:backward(input, output)
        end

        dl_dpred:div(batch_l)
        local dl_dtarget = generator:backward(preds[t], dl_dpred)

        local rnn_state_dec_pred_idx = #drnn_state_dec
        if opt.guided_alignment == 1 then
          rnn_state_dec_pred_idx = #drnn_state_dec-1
          drnn_state_dec[#drnn_state_dec]:add(drnn_state_attn)
        end
        drnn_state_dec[rnn_state_dec_pred_idx]:add(dl_dtarget)

        local decoder_input
        if opt.attn == 1 then
          decoder_input = {target[t], context, table.unpack(rnn_state_dec[t-1])}
        else
          decoder_input = {target[t], context[{{}, source_l}], table.unpack(rnn_state_dec[t-1])}
        end
        local dlst = decoder_clones[t]:backward(decoder_input, drnn_state_dec)
        -- accumulate encoder/decoder grads
        if opt.attn == 1 then
          encoder_grads:add(dlst[2])
          if opt.brnn == 1 then
            encoder_bwd_grads:add(dlst[2])
          end
        else
          encoder_grads[{{}, source_l}]:add(dlst[2])
          if opt.brnn == 1 then
            encoder_bwd_grads[{{}, 1}]:add(dlst[2])
          end
        end

        drnn_state_dec[rnn_state_dec_pred_idx]:zero()
        if opt.guided_alignment == 1 then
          drnn_state_dec[#drnn_state_dec]:zero()
        end
        if opt.input_feed == 1 then
          drnn_state_dec[rnn_state_dec_pred_idx]:add(dlst[3])
        end
        for j = dec_offset, #dlst do
          drnn_state_dec[j-dec_offset+1]:copy(dlst[j])
        end
      end
      word_vec_layers[2].gradWeight[1]:zero()
      if opt.fix_word_vecs_dec == 1 then
        word_vec_layers[2].gradWeight:zero()
      end

      local grad_norm = 0
      grad_norm = grad_norm + grad_params[2]:norm()^2 + grad_params[3]:norm()^2

      -- backward prop encoder
      if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
        cutorch.setDevice(opt.gpuid)
        local encoder_grads2 = encoder_grad_proto2[{{1, batch_l}, {1, source_l}}]
        encoder_grads2:zero()
        encoder_grads2:copy(encoder_grads)
        encoder_grads = encoder_grads2 -- batch_l x source_l x rnn_size
      end

      local drnn_state_enc = reset_state(init_bwd_enc, batch_l)
      if opt.init_dec == 1 then
        for L = 1, opt.num_layers do
          drnn_state_enc[L*2-1]:copy(drnn_state_dec[L*2-1])
          drnn_state_enc[L*2]:copy(drnn_state_dec[L*2])
        end
      end

      for t = source_l, 1, -1 do
        local encoder_input = {source[t]}
        if data.num_source_features > 0 then
          append_table(encoder_input, source_features[t])
        end
        append_table(encoder_input, rnn_state_enc[t-1])
        if opt.attn == 1 then
          drnn_state_enc[#drnn_state_enc]:add(encoder_grads[{{},t}])
        else
          if t == source_l then
            drnn_state_enc[#drnn_state_enc]:add(encoder_grads[{{},t}])
          end
        end
        local dlst = encoder_clones[t]:backward(encoder_input, drnn_state_enc)
        for j = 1, #drnn_state_enc do
          drnn_state_enc[j]:copy(dlst[j+1+data.num_source_features])
        end
      end

      if opt.brnn == 1 then
        local drnn_state_enc = reset_state(init_bwd_enc, batch_l)
        if opt.init_dec == 1 then
          for L = 1, opt.num_layers do
            drnn_state_enc[L*2-1]:copy(drnn_state_dec[L*2-1])
            drnn_state_enc[L*2]:copy(drnn_state_dec[L*2])
          end
        end
        for t = 1, source_l do
          local encoder_input = {source[t]}
          if data.num_source_features > 0 then
            append_table(encoder_input, source_features[t])
          end
          append_table(encoder_input, rnn_state_enc_bwd[t+1])
          if opt.attn == 1 then
            drnn_state_enc[#drnn_state_enc]:add(encoder_bwd_grads[{{},t}])
          else
            if t == 1 then
              drnn_state_enc[#drnn_state_enc]:add(encoder_bwd_grads[{{},t}])
            end
          end
          local dlst = encoder_bwd_clones[t]:backward(encoder_input, drnn_state_enc)
          for j = 1, #drnn_state_enc do
            drnn_state_enc[j]:copy(dlst[j+1+data.num_source_features])
          end
        end
      end

      word_vec_layers[1].gradWeight[1]:zero()
      if opt.fix_word_vecs_enc == 1 then
        word_vec_layers[1].gradWeight:zero()
      end

      grad_norm = grad_norm + grad_params[1]:norm()^2
      if opt.brnn == 1 then
        grad_norm = grad_norm + grad_params[4]:norm()^2
      end
      grad_norm = grad_norm^0.5
      if opt.brnn == 1 then
        word_vec_layers[1].gradWeight:add(word_vec_layers[3].gradWeight)
        if opt.use_chars_enc == 1 then
          for j = 1, charcnn_offset do
            charcnn_grad_layers[j]:add(charcnn_grad_layers[j+charcnn_offset])
          end
        end
      end
      -- Shrink norm and update params
      local param_norm = 0
      local shrinkage = opt.max_grad_norm / grad_norm
      for j = 1, #grad_params do
        if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
          if j == 1 then
            cutorch.setDevice(opt.gpuid)
          else
            cutorch.setDevice(opt.gpuid2)
          end
        end
        if shrinkage < 1 then
          grad_params[j]:mul(shrinkage)
        end
        if opt.optim == 'adagrad' then
          adagrad_step(params[j], grad_params[j], layer_etas[j], optStates[j])
        elseif opt.optim == 'adadelta' then
          adadelta_step(params[j], grad_params[j], layer_etas[j], optStates[j])
        elseif opt.optim == 'adam' then
          adam_step(params[j], grad_params[j], layer_etas[j], optStates[j])
        else
          params[j]:add(grad_params[j]:mul(-opt.learning_rate))
        end
        param_norm = param_norm + params[j]:norm()^2
      end
      param_norm = param_norm^0.5
      if opt.brnn == 1 then
        word_vec_layers[3].weight:copy(word_vec_layers[1].weight)
        if opt.use_chars_enc == 1 then
          for j = 1, charcnn_offset do
            charcnn_layers[j+charcnn_offset]:copy(charcnn_layers[j])
          end
        end
      end

      -- Bookkeeping
      num_words_target = num_words_target + batch_l*target_l
      num_words_source = num_words_source + batch_l*source_l
      train_nonzeros = train_nonzeros + nonzeros
      train_loss = train_loss + loss*batch_l
      if opt.guided_alignment == 1 then
        train_loss_cll = train_loss_cll + loss_cll*batch_l
      end
      local time_taken = timer:time().real - start_time
      if i % opt.print_every == 0 then
        local stats = string.format('Epoch: %d, Batch: %d/%d, Batch size: %d, LR: %.4f, ',
          epoch, i, data:size(), batch_l, opt.learning_rate)
        if opt.guided_alignment == 1 then
          stats = stats .. string.format('PPL: %.2f, PPL_CLL: %.2f, |Param|: %.2f, |GParam|: %.2f, ',
            math.exp(train_loss/train_nonzeros), math.exp(train_loss_cll/train_nonzeros), param_norm, grad_norm)
        else
          stats = stats .. string.format('PPL: %.2f, |Param|: %.2f, |GParam|: %.2f, ',
            math.exp(train_loss/train_nonzeros), param_norm, grad_norm)
        end
        stats = stats .. string.format('Training: %d/%d/%d total/source/target tokens/sec',
          (num_words_target+num_words_source) / time_taken,
          num_words_source / time_taken,
          num_words_target / time_taken)
        print(stats)
      end
      if i % 200 == 0 then
        collectgarbage()
      end
    end
    if opt.guided_alignment == 1 then
        return train_loss, train_nonzeros, train_loss_cll
    else
      return train_loss, train_nonzeros
    end
  end

  local total_loss, total_nonzeros, batch_loss, batch_nonzeros, total_loss_cll, batch_loss_cll
  for epoch = opt.start_epoch, opt.epochs do
    generator:training()
    if opt.num_shards > 0 then
      total_loss = 0
      total_nonzeros = 0
      local shard_order = torch.randperm(opt.num_shards)
      for s = 1, opt.num_shards do
        local fn = train_data .. '.' .. shard_order[s] .. '.hdf5'
        print('loading shard #' .. shard_order[s])
        local shard_data = data.new(opt, fn)
        if opt.guided_alignment == 1 then
          batch_loss, batch_nonzeros, batch_loss_cll = train_batch(shard_data, epoch)
          total_loss_cll = total_loss_cll + batch_loss_cll
        else
          batch_loss, batch_nonzeros = train_batch(shard_data, epoch)
        end
        total_loss = total_loss + batch_loss
        total_nonzeros = total_nonzeros + batch_nonzeros
      end
    else
      if opt.guided_alignment == 1 then
        total_loss, total_nonzeros, total_loss_cll = train_batch(train_data, epoch)
      else
        total_loss, total_nonzeros = train_batch(train_data, epoch)
      end
    end
    local train_score = math.exp(total_loss/total_nonzeros)
    print('Train', train_score)
    opt.train_perf[#opt.train_perf + 1] = train_score
    local score = eval(valid_data)
    opt.val_perf[#opt.val_perf + 1] = score
    if opt.optim == 'sgd' then --only decay with SGD
      decay_lr(epoch)
    end
    if opt.guided_alignment == 1 then
      opt.guided_alignment_weight = opt.guided_alignment_weight * opt.guided_alignment_decay
      criterion.weights[1] = 1-opt.guided_alignment_weight
      criterion.weights[2] = opt.guided_alignment_weight
    end
    -- clean and save models
    local savefile = string.format('%s_epoch%.2f_%.2f.t7', opt.savefile, epoch, score)
    if epoch % opt.save_every == 0 then
      print('saving checkpoint to ' .. savefile)
      clean_layer(generator)
      if opt.brnn == 0 then
        torch.save(savefile, {{encoder, decoder, generator}, opt})
      else
        torch.save(savefile, {{encoder, decoder, generator, encoder_bwd}, opt})
      end
    end
  end
  -- save final model
  local savefile = string.format('%s_final.t7', opt.savefile)
  clean_layer(generator)
  print('saving final model to ' .. savefile)
  if opt.brnn == 0 then
    torch.save(savefile, {{encoder:double(), decoder:double(), generator:double()}, opt})
  else
    torch.save(savefile, {{encoder:double(), decoder:double(), generator:double(),
          encoder_bwd:double()}, opt})
  end
end

function eval(data)
  encoder_clones[1]:evaluate()
  decoder_clones[1]:evaluate() -- just need one clone
  generator:evaluate()
  if opt.brnn == 1 then
    encoder_bwd_clones[1]:evaluate()
  end

  local nll = 0
  local nll_cll = 0
  local total = 0
  for i = 1, data:size() do
    local d = data[i]
    local target, target_out, nonzeros, source = d[1], d[2], d[3], d[4]
    local batch_l, target_l, source_l = d[5], d[6], d[7]
    local source_features = d[9]
    local alignment = d[10]
    local norm_alignment
    if opt.guided_alignment == 1 then
      replicator=nn.Replicate(alignment:size(2),2)
      if opt.gpuid >= 0 then
        cutorch.setDevice(opt.gpuid)
        if opt.gpuid2 >= 0 then -- alignment is in the 2nd GPU
          cutorch.setDevice(opt.gpuid2)
        end
        replicator = replicator:cuda()
      end
      norm_alignment = torch.cdiv(alignment, replicator:forward(torch.sum(alignment,2):squeeze(2)))
      norm_alignment[norm_alignment:ne(norm_alignment)] = 0
    end

    if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
      cutorch.setDevice(opt.gpuid)
    end
    local rnn_state_enc = reset_state(init_fwd_enc, batch_l)
    local context = context_proto[{{1, batch_l}, {1, source_l}}]
    -- forward prop encoder
    for t = 1, source_l do
      local encoder_input = {source[t]}
      if data.num_source_features > 0 then
        append_table(encoder_input, source_features[t])
      end
      append_table(encoder_input, rnn_state_enc)
      local out = encoder_clones[1]:forward(encoder_input)
      rnn_state_enc = out
      context[{{},t}]:copy(out[#out])
    end

    if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
      cutorch.setDevice(opt.gpuid2)
      local context2 = context_proto2[{{1, batch_l}, {1, source_l}}]
      context2:copy(context)
      context = context2
    end

    local rnn_state_dec = reset_state(init_fwd_dec, batch_l)
    if opt.init_dec == 1 then
      for L = 1, opt.num_layers do
        rnn_state_dec[L*2-1+opt.input_feed]:copy(rnn_state_enc[L*2-1])
        rnn_state_dec[L*2+opt.input_feed]:copy(rnn_state_enc[L*2])
      end
    end

    if opt.brnn == 1 then
      local rnn_state_enc = reset_state(init_fwd_enc, batch_l)
      for t = source_l, 1, -1 do
        local encoder_input = {source[t]}
        if data.num_source_features > 0 then
          append_table(encoder_input, source_features[t])
        end
        append_table(encoder_input, rnn_state_enc)
        local out = encoder_bwd_clones[1]:forward(encoder_input)
        rnn_state_enc = out
        context[{{},t}]:add(out[#out])
      end
      if opt.init_dec == 1 then
        for L = 1, opt.num_layers do
          rnn_state_dec[L*2-1+opt.input_feed]:add(rnn_state_enc[L*2-1])
          rnn_state_dec[L*2+opt.input_feed]:add(rnn_state_enc[L*2])
        end
      end
    end

    local loss = 0
    local loss_cll = 0
    local attn_outputs = {}
    for t = 1, target_l do
      local decoder_input
      if opt.attn == 1 then
        decoder_input = {target[t], context, table.unpack(rnn_state_dec)}
      else
        decoder_input = {target[t], context[{{},source_l}], table.unpack(rnn_state_dec)}
      end
      local out = decoder_clones[1]:forward(decoder_input)

      local out_pred_idx = #out
      if opt.guided_alignment == 1 then
        out_pred_idx = #out-1
        table.insert(attn_outputs, out[#out])
      end

      rnn_state_dec = {}
      if opt.input_feed == 1 then
        table.insert(rnn_state_dec, out[out_pred_idx])
      end
      for j = 1, out_pred_idx-1 do
        table.insert(rnn_state_dec, out[j])
      end
      local pred = generator:forward(out[out_pred_idx])

      local input = pred
      local output = target_out[t]
      if opt.guided_alignment == 1 then
        input={input, attn_outputs[t]}
        output={output, norm_alignment[{{},{},t}]}
      end

      loss = loss + criterion:forward(input, output)

      if opt.guided_alignment == 1 then
        loss_cll = loss_cll + cll_criterion:forward(input[1], output[1])
      end
    end
    nll = nll + loss
    if opt.guided_alignment == 1 then
      nll_cll = nll_cll + loss_cll
    end
    total = total + nonzeros
  end
  local valid = math.exp(nll / total)
  print("Valid", valid)
  if opt.guided_alignment == 1 then
    local valid_cll = math.exp(nll_cll / total)
    print("Valid_cll", valid_cll)
  end
  collectgarbage()
  return valid
end

function get_layer(layer)
  if layer.name ~= nil then
    if layer.name == 'word_vecs_dec' then
      table.insert(word_vec_layers, layer)
    elseif layer.name == 'word_vecs_enc' then
      table.insert(word_vec_layers, layer)
    elseif layer.name == 'charcnn_enc' or layer.name == 'mlp_enc' then
      local p, gp = layer:parameters()
      for i = 1, #p do
        table.insert(charcnn_layers, p[i])
        table.insert(charcnn_grad_layers, gp[i])
      end
    end
  end
end

function main()
  -- parse input params
  opt = cmd:parse(arg)

  torch.manualSeed(opt.seed)

  if opt.gpuid >= 0 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    if opt.gpuid2 >= 0 then
      print('using CUDA on second GPU ' .. opt.gpuid2 .. '...')
    end
    require 'cutorch'
    require 'cunn'
    if opt.cudnn == 1 then
      print('loading cudnn...')
      require 'cudnn'
    end
    cutorch.setDevice(opt.gpuid)
    cutorch.manualSeed(opt.seed)
  end

  -- Create the data loader class.
  print('loading data...')
  if opt.num_shards == 0 then
    train_data = data.new(opt, opt.data_file)
  else
    train_data = opt.data_file
  end

  valid_data = data.new(opt, opt.val_data_file)
  print('done!')
  print(string.format('Source vocab size: %d, Target vocab size: %d',
      valid_data.source_size, valid_data.target_size))
  opt.max_sent_l_src = valid_data.source:size(2)
  opt.max_sent_l_targ = valid_data.target:size(2)
  opt.max_sent_l = math.max(opt.max_sent_l_src, opt.max_sent_l_targ)
  if opt.max_batch_l == '' then
    opt.max_batch_l = valid_data.batch_l:max()
  end

  if opt.use_chars_enc == 1 or opt.use_chars_dec == 1 then
    opt.max_word_l = valid_data.char_length
  end
  print(string.format('Source max sent len: %d, Target max sent len: %d',
      valid_data.source:size(2), valid_data.target:size(2)))

  print(string.format('Number of additional features on source side: %d', valid_data.num_source_features))

  -- Enable memory preallocation - see memory.lua
  preallocateMemory(opt.prealloc)

  -- Build model
  if opt.train_from:len() == 0 then
    encoder = make_lstm(valid_data, opt, 'enc', opt.use_chars_enc)
    decoder = make_lstm(valid_data, opt, 'dec', opt.use_chars_dec)
    generator, criterion = make_generator(valid_data, opt)
    if opt.brnn == 1 then
      encoder_bwd = make_lstm(valid_data, opt, 'enc', opt.use_chars_enc)
    end
  else
    assert(path.exists(opt.train_from), 'checkpoint path invalid')
    print('loading ' .. opt.train_from .. '...')
    local checkpoint = torch.load(opt.train_from)
    local model, model_opt = checkpoint[1], checkpoint[2]
    opt.num_layers = model_opt.num_layers
    opt.rnn_size = model_opt.rnn_size
    opt.input_feed = model_opt.input_feed
    opt.attn = model_opt.attn
    opt.brnn = model_opt.brnn
    encoder = model[1]
    decoder = model[2]
    generator = model[3]
    if model_opt.brnn == 1 then
      encoder_bwd = model[4]
    end
    _, criterion = make_generator(valid_data, opt)
  end

  if opt.guided_alignment == 1 then
    cll_criterion = criterion
    criterion = nn.ParallelCriterion()
    criterion:add(cll_criterion, (1-opt.guided_alignment_weight))
    -- sum of alignment weight reconstruction loss over all input/output pair; averaged
    criterion:add(nn.MSECriterion(), opt.guided_alignment_weight)
  end

  layers = {encoder, decoder, generator}
  if opt.brnn == 1 then
    table.insert(layers, encoder_bwd)
  end

  if opt.optim ~= 'sgd' then
    layer_etas = {}
    optStates = {}
    for i = 1, #layers do
      layer_etas[i] = opt.learning_rate -- can have layer-specific lr, if desired
      optStates[i] = {}
    end
  end

  if opt.gpuid >= 0 then
    for i = 1, #layers do
      if opt.gpuid2 >= 0 then
        if i == 1 or i == 4 then
          cutorch.setDevice(opt.gpuid) --encoder on gpu1
        else
          cutorch.setDevice(opt.gpuid2) --decoder/generator on gpu2
        end
      end
      layers[i]:cuda()
    end
    if opt.gpuid2 >= 0 then
      cutorch.setDevice(opt.gpuid2) --criterion on gpu2
    end
    criterion:cuda()
  end

  -- these layers will be manipulated during training
  word_vec_layers = {}
  if opt.use_chars_enc == 1 then
    charcnn_layers = {}
    charcnn_grad_layers = {}
  end
  encoder:apply(get_layer)
  decoder:apply(get_layer)
  if opt.brnn == 1 then
    if opt.use_chars_enc == 1 then
      charcnn_offset = #charcnn_layers
    end
    encoder_bwd:apply(get_layer)
  end
  train(train_data, valid_data)
end

main()
