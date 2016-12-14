 --[[ Model, adapted from https://github.com/harvardnlp/seq2seq-attn/blob/master/train.lua
--]]
require 'nn'
require 'cudnn'
require 'optim'
require 'paths'
require('../../../../onmt/utils')
require('../../../../onmt/modules')
package.path = package.path .. ';src/?.lua' .. ';src/utils/?.lua' .. ';src/model/?.lua' .. ';src/optim/?.lua'
require 'cnn'
require 'LSTM'
require 'output_projector'
require 'criterion'
require 'model_utils'
require 'optim_adadelta'
require 'optim_sgd'
require 'memory'

local model = torch.class('Model')

--[[ Args: 
-- config.load_model
-- config.model_dir
-- config.dropout
-- config.encoder_num_hidden
-- config.encoder_num_layers
-- config.decoder_num_layers
-- config.target_vocab_size
-- config.target_embedding_size
-- config.max_encoder_l_w
-- config.max_decoder_l
-- config.input_feed
-- config.batch_size
--]]

-- init
function model:__init()
    if logging ~= nil then
        log = function(msg) logging:info(msg) end
    else
        log = print
    end
end

-- load model from model_path
function model:load(model_path, config)
    config = config or {}

    -- Build model

    assert(paths.filep(model_path), string.format('Model %s does not exist!', model_path))

    local checkpoint = torch.load(model_path)
    local model, model_config = checkpoint[1], checkpoint[2]
    preallocateMemory(model_config.prealloc)
    self.cnn_model = model[1]:double()
    self.encoder = model[2]:double()
    self.decoder = model[3]:double()      
    self.pos_embedding_fw = model[6]:double()
    self.pos_embedding_bw = model[7]:double()
    self.global_step = checkpoint[3]
    self.optim_state = checkpoint[4]
    id2vocab = checkpoint[5]

    -- Load model structure parameters
    self.cnn_feature_size = 512
    self.dropout = model_config.dropout
    self.encoder_num_hidden = model_config.encoder_num_hidden
    self.encoder_num_layers = model_config.encoder_num_layers
    self.decoder_num_hidden = self.encoder_num_hidden * 2
    self.decoder_num_layers = model_config.decoder_num_layers
    self.target_vocab_size = #id2vocab+4
    self.target_embedding_size = model_config.target_embedding_size
    self.input_feed = model_config.input_feed
    self.prealloc = model_config.prealloc

    self.max_encoder_l_w = config.max_encoder_l_w or model_config.max_encoder_l_w
    self.max_encoder_l_h = config.max_encoder_l_h or model_config.max_encoder_l_h
    self.max_decoder_l = config.max_decoder_l or model_config.max_decoder_l
    self.batch_size = config.batch_size or model_config.batch_size

    if config.max_encoder_l_h > model_config.max_encoder_l_h then
        local pos_embedding_fw = nn.Sequential():add(nn.LookupTable(self.max_encoder_l_h,self.encoder_num_layers*self.encoder_num_hidden*2))
        local pos_embedding_bw = nn.Sequential():add(nn.LookupTable(self.max_encoder_l_h, self.encoder_num_layers*self.encoder_num_hidden*2))
        for i = 1, self.max_encoder_l_h do
            local j = math.min(i, model_config.max_encoder_l_h)
            pos_embedding_fw:get(1).weight[i] = self.pos_embedding_fw:get(1).weight[j]
            pos_embedding_bw:get(1).weight[i] = self.pos_embedding_bw:get(1).weight[j]
        end
        self.pos_embedding_fw = pos_embedding_fw
        self.pos_embedding_bw = pos_embedding_bw
    end
    self:_build()
end

-- create model with fresh parameters
function model:create(config)
    self.cnn_feature_size = 512
    self.dropout = config.dropout
    self.encoder_num_hidden = config.encoder_num_hidden
    self.encoder_num_layers = config.encoder_num_layers
    self.decoder_num_hidden = config.encoder_num_hidden * 2
    self.decoder_num_layers = config.decoder_num_layers
    self.target_vocab_size = config.target_vocab_size
    self.target_embedding_size = config.target_embedding_size
    self.max_encoder_l_w = config.max_encoder_l_w
    self.max_encoder_l_h = config.max_encoder_l_h
    self.max_decoder_l = config.max_decoder_l
    self.input_feed = config.input_feed
    self.batch_size = config.batch_size
    self.prealloc = config.prealloc
    preallocateMemory(config.prealloc)

    self.pos_embedding_fw = nn.Sequential():add(nn.LookupTable(self.max_encoder_l_h,self.encoder_num_layers*self.encoder_num_hidden*2))
    self.pos_embedding_bw = nn.Sequential():add(nn.LookupTable(self.max_encoder_l_h, self.encoder_num_layers*self.encoder_num_hidden*2))
    -- CNN model, input size: (batch_size, 1, 32, width), output size: (batch_size, sequence_length, 512)
    self.cnn_model = createCNNModel()
    -- biLSTM encoder
    local rnn = onmt.LSTM.new(self.encoder_num_layers, self.cnn_feature_size, self.encoder_num_hidden, self.dropout)
    local input_network = nn.Sequential():add(nn.Identity())
    self.encoder = onmt.BiEncoder.new(input_network, rnn, 'concat')

    -- decoder
    local input_network = onmt.WordEmbedding.new(self.target_vocab_size, self.target_embedding_size)
    local input_size = self.target_embedding_size
    if self.input_feed then
        input_size = input_size + self.decoder_num_hidden
    end
    local rnn = onmt.LSTM.new(self.decoder_num_layers, input_size, self.decoder_num_hidden, self.dropout)
    local generator = onmt.Generator.new(self.decoder_num_hidden, self.target_vocab_size)
    self.decoder = onmt.Decoder.new(input_network, rnn, generator, self.input_feed)

    self.global_step = 0
    self._init = true

    self.optim_state = {}
    self.optim_state.learningRate = config.learning_rate
    self:_build()
end

-- build
function model:_build()
    log(string.format('cnn_featuer_size: %d', self.cnn_feature_size))
    log(string.format('dropout: %f', self.dropout))
    log(string.format('encoder_num_hidden: %d', self.encoder_num_hidden))
    log(string.format('encoder_num_layers: %d', self.encoder_num_layers))
    log(string.format('decoder_num_hidden: %d', self.decoder_num_hidden))
    log(string.format('decoder_num_layers: %d', self.decoder_num_layers))
    log(string.format('target_vocab_size: %d', self.target_vocab_size))
    log(string.format('target_embedding_size: %d', self.target_embedding_size))
    log(string.format('max_encoder_l_w: %d', self.max_encoder_l_w))
    log(string.format('max_encoder_l_h: %d', self.max_encoder_l_h))
    log(string.format('max_decoder_l: %d', self.max_decoder_l))
    log(string.format('input_feed: %s', self.input_feed))
    log(string.format('batch_size: %d', self.batch_size))
    log(string.format('prealloc: %s', self.prealloc))


    self.config = {}
    self.config.dropout = self.dropout
    self.config.encoder_num_hidden = self.encoder_num_hidden
    self.config.encoder_num_layers = self.encoder_num_layers
    self.config.decoder_num_hidden = self.decoder_num_hidden
    self.config.decoder_num_layers = self.decoder_num_layers
    self.config.target_vocab_size = self.target_vocab_size
    self.config.target_embedding_size = self.target_embedding_size
    self.config.max_encoder_l_w = self.max_encoder_l_w
    self.config.max_encoder_l_h = self.max_encoder_l_h
    self.config.max_decoder_l = self.max_decoder_l
    self.config.input_feed = self.input_feed
    self.config.batch_size = self.batch_size
    self.config.prealloc = self.prealloc


    if self.optim_state == nil then
        self.optim_state = {}
    end
    self.criterion = createCriterion(self.target_vocab_size)

    -- convert to cuda if use gpu
    self.layers = {self.cnn_model, self.encoder, self.decoder, self.pos_embedding_fw, self.pos_embedding_bw}
    for i = 1, #self.layers do
        utils.Cuda.convert(self.layers[i])
    end
    localize(self.criterion)

    self.context_proto = localize(torch.zeros(self.batch_size, self.max_encoder_l_w*self.max_encoder_l_h, 2*self.encoder_num_hidden))
    self.cnn_grad_proto = localize(torch.zeros(self.max_encoder_l_h, self.batch_size, self.max_encoder_l_w, self.cnn_feature_size))

    local num_params = 0
    self.params, self.grad_params = {}, {}
    for i = 1, #self.layers do
        local p, gp = self.layers[i]:getParameters()
        if self._init then
            p:uniform(-0.05,0.05)
        end
        num_params = num_params + p:size(1)
        self.params[i] = p
        self.grad_params[i] = gp
    end
    log(string.format('Number of parameters: %d', num_params))

    self.init_beam = false
    self.visualize = false
    collectgarbage()
end

-- one step 
function model:step(batch, forward_only, beam_size, trie)
    if forward_only then
        self.val_batch_size = self.batch_size
        beam_size = beam_size or 1 -- default argmax
        beam_size = math.min(beam_size, self.target_vocab_size)
        if not self.init_beam then
            self.init_beam = true
            local beam_decoder_h_init = localize(torch.zeros(self.val_batch_size*beam_size, self.decoder_num_hidden))
            self.beam_scores = localize(torch.zeros(self.val_batch_size, beam_size))
            self.current_indices_history = {}
            self.beam_parents_history = {}
            self.beam_init_fwd_dec = {}
            if self.input_feed then
                table.insert(self.beam_init_fwd_dec, beam_decoder_h_init:clone())
            end
            for L = 1, self.decoder_num_layers do
                table.insert(self.beam_init_fwd_dec, beam_decoder_h_init:clone()) -- memory cell
                table.insert(self.beam_init_fwd_dec, beam_decoder_h_init:clone()) -- hidden state
            end
            self.trie_locations = {}
        else
            self.beam_scores:zero()
            self.current_indices_history = {}
            self.beam_parents_history = {}
            self.trie_locations = {}
        end
    else
        if self.init_beam then
            self.init_beam = false
            self.trie_locations = {}
            self.beam_init_fwd_dec = {}
            self.current_indices_history = {}
            self.beam_parents_history = {}
            self.trie_locations = {}
            self.beam_scores = nil
            collectgarbage()
        end
    end
    local input_batch = utils.Cuda.convert(batch[1])
    local target_batch = utils.Cuda.convert(batch[2])
    local target_eval_batch = utils.Cuda.convert(batch[3])
    local num_nonzeros = batch[4]
    local img_paths
    if self.visualize then
        img_paths = batch[5]
    end

    local batch_size = input_batch:size()[1]
    local target_l = target_batch:size()[2]

    assert(target_l <= self.max_decoder_l, string.format('max_decoder_l (%d) < target_l (%d)!', self.max_decoder_l, target_l))
    -- if forward only, then re-generate the target batch
    if forward_only then
        local target_batch_new = localize(torch.IntTensor(batch_size, self.max_decoder_l)):fill(1)
        target_batch_new[{{1,batch_size}, {1,target_l}}]:copy(target_batch)
        target_batch = target_batch_new
        local target_eval_batch_new = localize(torch.IntTensor(batch_size, self.max_decoder_l)):fill(1)
        target_eval_batch_new[{{1,batch_size}, {1,target_l}}]:copy(target_eval_batch)
        target_eval_batch = target_eval_batch_new
        target_l = self.max_decoder_l
    end

    if not forward_only then
        self.cnn_model:training()
        self.encoder:training()
        self.decoder:training()
        self.pos_embedding_fw:training()
        self.pos_embedding_bw:training()
    else
        self.cnn_model:evaluate()
        self.encoder:evaluate()
        self.decoder:evaluate()
        self.pos_embedding_fw:evaluate()
        self.pos_embedding_bw:evaluate()
    end

    local feval = function(p) --cut off when evaluate
        target = target_batch:transpose(1,2)
        target_eval = target_eval_batch:transpose(1,2)
        local cnn_output_list = self.cnn_model:forward(input_batch) -- list of (batch_size, W, 512)
        local counter = 1
        local imgH = #cnn_output_list
        local source_l = cnn_output_list[1]:size()[2]
        local context = self.context_proto[{{1, batch_size}, {1, source_l*imgH}}] --batch_size, source_l*imgH, 512
        local dec_batch = Batch():set_decoder(target, target_eval, context:size(2))
        for i = 1, imgH do
            local pos = localize(torch.zeros(batch_size)):fill(i)
            local pos_embedding_fw  = self.pos_embedding_fw:forward(pos):view(1,batch_size,-1) --1,batch_size,512
            local pos_embedding_bw  = self.pos_embedding_bw:forward(pos):view(1,batch_size,-1)
            local cnn_output = cnn_output_list[i] --batch_size, imgW, 512
            source = cnn_output:transpose(1,2) -- imgW,batch_size,512
            local input = torch.cat(pos_embedding_fw, source, 1)
            input = torch.cat(input, pos_embedding_bw, 1)
            local batch = Batch():set_encoder(input)
            local _, row_context = self.encoder:forward(batch)
            for t = 1, source_l do
                counter = (i-1)*source_l + t
                context[{{}, counter, {}}]:copy(row_context[{{}, t+1, {}}])
            end
        end
        local preds = {}
        local indices
        local rnn_state_dec
        local dec_outputs
        -- forward_only == true, beam search
        if forward_only then
            local beam_replicate = function(hidden_state)
                if hidden_state:dim() == 1 then
                    local batch_size = hidden_state:size()[1]
                    if not hidden_state:isContiguous() then
                        hidden_state = hidden_state:contiguous()
                    end
                    local temp_state = hidden_state:view(batch_size, 1):expand(batch_size, beam_size)
                    if not temp_state:isContiguous() then
                        temp_state = temp_state:contiguous()
                    end
                    return temp_state:view(-1)
                elseif hidden_state:dim() == 2 then
                    local batch_size = hidden_state:size()[1]
                    local num_hidden = hidden_state:size()[2]
                    if not hidden_state:isContiguous() then
                        hidden_state = hidden_state:contiguous()
                    end
                    local temp_state = hidden_state:view(batch_size, 1, num_hidden):expand(batch_size, beam_size, num_hidden)
                    if not temp_state:isContiguous() then
                        temp_state = temp_state:contiguous()
                    end
                    return temp_state:view(batch_size*beam_size, num_hidden)
                elseif hidden_state:dim() == 3 then
                    local batch_size = hidden_state:size()[1]
                    local source_l = hidden_state:size()[2]
                    local num_hidden = hidden_state:size()[3]
                    if not hidden_state:isContiguous() then
                        hidden_state = hidden_state:contiguous()
                    end
                    local temp_state = hidden_state:view(batch_size, 1, source_l, num_hidden):expand(batch_size, beam_size, source_l, num_hidden)
                    if not temp_state:isContiguous() then
                        temp_state = temp_state:contiguous()
                    end
                    return temp_state:view(batch_size*beam_size, source_l, num_hidden)
                else
                    assert(false, 'does not support ndim except for 2 and 3')
                end
            end
            rnn_state_dec = reset_state(self.beam_init_fwd_dec, batch_size, 0)
            local beam_context = beam_replicate(context)
            local decoder_input
            local beam_input
            local dec_states = utils.Tensor.initTensorTable(self.decoder.args.numEffectiveLayers,
                                                    localize(torch.Tensor()),
                                                    { batch_size, self.decoder.args.rnnSize })
            for i = 1, #dec_states do
                dec_states[i]:zero()
            end
            local dec_out = nil
            --dec_outputs = self.decoder:forward(dec_batch, statesProto, context)
            for t = 1, target_l do
                local dec_context
                if t == 1 then
                    beam_input = target[t]
                    dec_context = context
                else
                    dec_context = beam_context
                end
                dec_out, dec_states = self.decoder:forward_one(beam_input, dec_states, dec_context, dec_out, t)
                local probs = self.decoder.generator:forward(dec_out)[1] -- t~=0, batch_size*beam_size, vocab_size; t=0, batch_size,vocab_size
                local current_indices, raw_indices
                local beam_parents
                if t == 1 then
                    -- probs batch_size, vocab_size
                    self.beam_scores, raw_indices = probs:topk(beam_size, true)
                    raw_indices = localize(raw_indices:double())
                    current_indices = raw_indices
                else
                    -- batch_size*beam_size, vocab_size
                    probs:select(2,1):maskedFill(beam_input:eq(1), 0) -- once padding or EOS encountered, stuck at that point
                    probs:select(2,1):maskedFill(beam_input:eq(3), 0)
                    local total_scores = (probs:view(batch_size, beam_size, self.target_vocab_size) + self.beam_scores[{{1,batch_size}, {}}]:view(batch_size, beam_size, 1):expand(batch_size, beam_size, self.target_vocab_size)):view(batch_size, beam_size*self.target_vocab_size) -- batch_size, beam_size * target_vocab_size
                    self.beam_scores, raw_indices = total_scores:topk(beam_size, true) --batch_size, beam_size
                    raw_indices = localize(raw_indices:double())
                    raw_indices:add(-1)
                    if use_cuda then
                        current_indices = raw_indices:double():fmod(self.target_vocab_size):cuda()+1 -- batch_size, beam_size for current vocab
                    else
                        current_indices = raw_indices:fmod(self.target_vocab_size)+1 -- batch_size, beam_size for current vocab
                    end
                end
                beam_parents = localize(raw_indices:int()/self.target_vocab_size+1) -- batch_size, beam_size for number of beam in each batch
                beam_input = current_indices:view(batch_size*beam_size)
                table.insert(self.current_indices_history, current_indices:clone())
                table.insert(self.beam_parents_history, beam_parents:clone())

                if self.input_feed then
                    if t == 1 then
                        dec_out = beam_replicate(dec_out)
                    end
                    dec_out = dec_out:index(1, beam_parents:view(-1)+localize(torch.range(0,(batch_size-1)*beam_size,beam_size):long()):contiguous():view(batch_size,1):expand(batch_size,beam_size):contiguous():view(-1))
                end
                for j = 1, #dec_states do
                    local out_j = dec_states[j] -- batch_size*beam_size, hidden_dim
                    if t == 1 then
                        out_j = beam_replicate(out_j)
                    end
                    dec_states[j] = out_j:index(1, beam_parents:view(-1)+localize(torch.range(0,(batch_size-1)*beam_size,beam_size):long()):contiguous():view(batch_size,1):expand(batch_size,beam_size):contiguous():view(-1))
                end
            end
        else -- forward_only == false
            -- set decoder states
            local statesProto = utils.Tensor.initTensorTable(self.decoder.args.numEffectiveLayers,
                                                    localize(torch.Tensor()),
                                                    { dec_batch.size, self.decoder.args.rnnSize })
            for i = 1, #statesProto do
                statesProto[i]:zero()
            end
            dec_outputs = self.decoder:forward(dec_batch, statesProto, context)
        end
        local loss, accuracy = 0.0, 0.0
        if forward_only then
            -- final decoding
            local labels = localize(torch.zeros(batch_size, target_l)):fill(1)
            local scores, indices = torch.max(self.beam_scores[{{1,batch_size},{}}], 2) -- batch_size, 1
            indices = localize(indices:double())
            scores = scores:view(-1) -- batch_size
            indices = indices:view(-1) -- batch_size
            local current_indices = self.current_indices_history[#self.current_indices_history]:view(-1):index(1,indices+localize(torch.range(0,(batch_size-1)*beam_size, beam_size):long())) --batch_size
            for t = target_l, 1, -1 do
                labels[{{1,batch_size}, t}]:copy(current_indices)
                indices = self.beam_parents_history[t]:view(-1):index(1,indices+localize(torch.range(0,(batch_size-1)*beam_size, beam_size):long())) --batch_size
                if t > 1 then
                    current_indices = self.current_indices_history[t-1]:view(-1):index(1,indices+localize(torch.range(0,(batch_size-1)*beam_size, beam_size):long())) --batch_size
                end
            end
            local word_err, labels_pred, labels_gold, labels_list_pred, labels_list_gold = evalHTMLErrRate(labels, target_eval_batch, self.visualize)
            accuracy = batch_size - word_err
            if self.visualize then
                -- get gold score
                local statesProto = utils.Tensor.initTensorTable(self.decoder.args.numEffectiveLayers,
                                                        localize(torch.Tensor()),
                                                        { dec_batch.size, self.decoder.args.rnnSize })
                for i = 1, #statesProto do
                    statesProto[i]:zero()
                end
                local gold_scores = self.decoder:compute_score(dec_batch, statesProto, context)
                -- use predictions to visualize attns
                local attn_probs = localize(torch.zeros(batch_size, target_l, source_l*imgH))
                local attn_positions_h = localize(torch.zeros(batch_size, target_l))
                local attn_positions_w = localize(torch.zeros(batch_size, target_l))
                local dec_states = utils.Tensor.initTensorTable(self.decoder.args.numEffectiveLayers,
                                                        localize(torch.Tensor()),
                                                        { batch_size, self.decoder.args.rnnSize })
                for i = 1, #dec_states do
                    dec_states[i]:zero()
                end
                local dec_out = nil
                --dec_outputs = self.decoder:forward(dec_batch, statesProto, context)
                for t = 1, target_l do
                    if t == 1 then
                        decoder_input = dec_batch:get_target_input(t)
                    else
                        decoder_input = labels[{{1,batch_size},t-1}]
                    end
                    dec_out, dec_states = self.decoder:forward_one(decoder_input(t), dec_states, context, dec_out, t)
                    local softmax_out = self.decoder.softmax_attn.output
                    -- print attn
                    attn_probs[{{}, t, {}}]:copy(softmax_out)
                    local _, attn_inds = torch.max(softmax_out, 2) --batch_size, 1
                    attn_inds = attn_inds:view(-1) --batch_size
                    for kk = 1, batch_size do
                        local counter = attn_inds[kk]
                        local p_i = math.floor((counter-1) / source_l) + 1
                        local p_t = counter-1 - (p_i-1)*source_l + 1
                        attn_positions_h[kk][t] = p_i
                        attn_positions_w[kk][t] = p_t
                        --print (string.format('%d, %d', p_i, p_t))
                    end
                    local pred = self.decoder.generator:forward(dec_out) --batch_size, vocab_size

                end
                for i = 1, #img_paths do
                    self.visualize_file:write(string.format('%s\t%s\t%s\t%f\t%f\t\n', img_paths[i], labels_gold[i], labels_pred[i], scores[i], gold_scores[i]))
                end
                self.visualize_file:flush()
            end
            -- get gold score
            dec_batch = Batch():set_decoder(target, target_eval, context:size(2))
            local statesProto = utils.Tensor.initTensorTable(self.decoder.args.numEffectiveLayers,
                                                    localize(torch.Tensor()),
                                                    { dec_batch.size, self.decoder.args.rnnSize })
            for i = 1, #statesProto do
                statesProto[i]:zero()
            end
            loss = self.decoder:compute_loss(dec_batch, statesProto, context, self.criterion)/batch_size
        else
            for i = 1, #self.grad_params do
                self.grad_params[i]:zero()
            end
            local _, grad_context, raw_loss = self.decoder:backward(dec_batch, dec_outputs, self.criterion)
            loss = raw_loss / batch_size
            grad_context = grad_context:contiguous():view(batch_size, imgH, source_l, -1)
            --grad_context: batch_size, imgH, source_l, 512
            local grad_padding = localize(torch.zeros(batch_size, imgH, 1, grad_context:size(4)))
            grad_context = torch.cat(grad_padding, grad_context, 3)
            grad_context = torch.cat(grad_context, grad_padding, 3)
            --grad_context: batch_size, imgH, source_l+2, 512
            --local _, row_context = self.encoder:forward(batch)
            --for t = 1, source_l do
            --    counter = (i-1)*source_l + t
            --    context[{{}, counter, {}}]:copy(row_context[{{}, t+1, {}}])
            --end
            local cnn_grad = self.cnn_grad_proto[{{1,imgH}, {1,batch_size}, {1,source_l}, {}}]
            -- forward directional encoder
            for i = 1, imgH do
                local cnn_output = cnn_output_list[i]
                source = cnn_output:transpose(1,2) -- 128,1,512
                assert (source_l == cnn_output:size()[2])
                local pos = localize(torch.zeros(batch_size)):fill(i)
                local pos_embedding_fw = self.pos_embedding_fw:forward(pos):view(1,batch_size,-1)
                local pos_embedding_bw = self.pos_embedding_bw:forward(pos):view(1,batch_size,-1)
                local input = torch.cat(pos_embedding_fw, source, 1)
                input = torch.cat(input, pos_embedding_bw, 1) --source_l+2, batch_size, hidden
                local batch = Batch():set_encoder(input)
                local _, row_context = self.encoder:forward(batch)
                local gradOutputsProto = utils.Tensor.initTensorTable(self.encoder.args.numEffectiveLayers,
                                                         localize(torch.Tensor()),
                                                         { batch.size, self.encoder.args.rnnSize*2 })
                local row_context_grad = self.encoder:backward(batch, gradOutputsProto, grad_context:select(2,i))
                -- source_l+2, batch_size, hidden
                for t = 1, source_l do
                    cnn_grad[{i, {}, t, {}}]:copy(row_context_grad[t])
                end
                self.pos_embedding_fw:backward(pos, row_context_grad[1])
                self.pos_embedding_bw:backward(pos, row_context_grad[source_l+2])
            end
            -- cnn
            local cnn_final_grad = cnn_grad:split(1, 1)
            for i = 1, #cnn_final_grad do
                cnn_final_grad[i] = cnn_final_grad[i]:contiguous():view(batch_size, source_l, -1)
            end
            self.cnn_model:backward(input_batch, cnn_final_grad)
            collectgarbage()
        end
        return loss, self.grad_params, {num_nonzeros, accuracy}
    end
    local optim_state = self.optim_state
    if not forward_only then
        local _, loss, stats = optim.sgd_list(feval, self.params, optim_state); loss = loss[1]
        return loss*batch_size, stats
    else
        local loss, _, stats = feval(self.params)
        return loss*batch_size, stats -- todo: accuracy
    end
end
-- Set visualize phase
function model:vis(output_dir)
    self.visualize = true
    self.visualize_path = paths.concat(output_dir, 'results.txt')
    self.visualize_attn_path = paths.concat(output_dir, 'results_attn.txt')
    local file, err = io.open(self.visualize_path, "w")
    local file_attn, err_attn = io.open(self.visualize_attn_path, "w")
    self.visualize_file = file
    self.visualize_attn_file = file_attn
    if err then 
        log(string.format('Error: visualize file %s cannot be created', self.visualize_path))
        self.visualize  = false
        self.visualize_file = nil
    elseif err_attn then
        log(string.format('Error: visualize attention file %s cannot be created', self.visualize_attn_path))
        self.visualize  = false
        self.visualize_attn_file = nil
    end
end
-- Save model to model_path
function model:save(model_path)
    for i = 1, #self.layers do
        self.layers[i]:clearState()
    end
    torch.save(model_path, {{self.cnn_model, self.encoder_fw, self.encoder_bw, self.decoder, self.output_projector, self.pos_embedding_fw, self.pos_embedding_bw}, self.config, self.global_step, self.optim_state, id2vocab})
end

function model:shutdown()
    if self.visualize_file then
        self.visualize_file:close()
    end
    if self.visualize_attn_file then
        self.visualize_attn_file:close()
    end
end
