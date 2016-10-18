require 'nn'
require 'string'
require 'hdf5'
require 'nngraph'

require 'models.lua'
require 'data.lua'
require 'util.lua'

stringx = require('pl.stringx')

cmd = torch.CmdLine()

-- file location
cmd:option('-model', 'seq2seq_lstm_attn.t7.', [[Path to model .t7 file]])
cmd:option('-src_file', '', [[Source sequence to decode (one line per sequence)]])
cmd:option('-targ_file', '', [[True target sequence]])
cmd:option('-output_file', 'pred.txt', [[Path to output the predictions (each line will be the
                                       decoded sequence]])
cmd:option('-src_dict', 'data/demo.src.dict', [[Path to source vocabulary (*.src.dict file)]])
cmd:option('-targ_dict', 'data/demo.targ.dict', [[Path to target vocabulary (*.targ.dict file)]])
cmd:option('-char_dict', 'data/demo.char.dict', [[If using chars, path to character 
                                                vocabulary (*.char.dict file)]])

-- beam search options
cmd:option('-max_sent_l', 250, [[Maximum sentence length. If any sequences in srcfile are longer
                               than this then it will error out]])
cmd:option('-replace_unk', 0, [[Replace the generated UNK tokens with the source token that 
                              had the highest attention weight. If srctarg_dict is provided, 
                              it will lookup the identified source token and give the corresponding 
                              target token. If it is not provided (or the identified source token
                              does not exist in the table) then it will copy the source token]])
cmd:option('-srctarg_dict', 'data/en-de.dict', [[Path to source-target dictionary to replace UNK 
                             tokens. See README.md for the format this file should be in]])
cmd:option('-gpuid', -1, [[ID of the GPU to use (-1 = use CPU)]])
cmd:option('-gpuid2', -1, [[Second GPU ID]])
cmd:option('-cudnn', 0, [[If using character model, this should be = 1 if the character model
                          was trained using cudnn]])
opt = cmd:parse(arg)

function copy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in pairs(orig) do
            copy[orig_key] = orig_value
        end
    else
        copy = orig
    end
    return copy
end


-- Convert a flat index to a row-column tuple.
function flat_to_rc(v, flat_index)
    local row = math.floor((flat_index - 1) / v:size(2)) + 1
    return row, (flat_index - 1) % v:size(2) + 1
end


function generate_beam(model, initial, max_sent_l, source, gold)
    --reset decoder initial states
    if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
        cutorch.setDevice(opt.gpuid)
    end
    local n = max_sent_l
    local source_l = math.min(source:size(1), opt.max_sent_l)

    local source_input
    if model_opt.use_chars_enc == 1 then
        source_input = source:view(source_l, 1, source:size(2)):contiguous()
    else
        source_input = source:view(source_l, 1)
    end

    local rnn_state_enc = {}
    for i = 1, #init_fwd_enc do
        table.insert(rnn_state_enc, init_fwd_enc[i]:zero())
    end
    local context = context_proto[{ {}, { 1, source_l } }]:clone() -- 1 x source_l x rnn_size

    print("ENCODER POS", saved_encoder_position)
    local position_tensor = torch.LongTensor({ saved_encoder_position })
    --initial write and then append
    if saved_encoder_position == 0 then
        encoderfile:write('offsets', position_tensor, offset_options)
    else
        encoderfile:append('offsets', position_tensor, offset_options)
    end

    for t = 1, source_l do
        local encoder_input = { source_input[t], table.unpack(rnn_state_enc) }
        local out = model[1]:forward(encoder_input)
        rnn_state_enc = out

        if saved_encoder_position == 0 then
            print("TEST")
            for k = 1, 2 * model_opt.num_layers do
                encoderfile:write("states" .. k, out[k], state_options)
            end
        else

            for k = 1, 2 * model_opt.num_layers do
                encoderfile:append("states" .. k, out[k], state_options)
            end
        end
        saved_encoder_position = saved_encoder_position + 1

        context[{ {}, t }]:copy(out[#out])
    end

    rnn_state_dec = {}
    for i = 1, #init_fwd_dec do
        table.insert(rnn_state_dec, init_fwd_dec[i]:zero())
    end

    if model_opt.init_dec == 1 then
        for L = 1, model_opt.num_layers do
            rnn_state_dec[L * 2 - 1 + model_opt.input_feed]:copy(rnn_state_enc[L * 2 - 1])
            rnn_state_dec[L * 2 + model_opt.input_feed]:copy(rnn_state_enc[L * 2])
        end
    end


    if model_opt.brnn == 1 then
        for i = 1, #rnn_state_enc do
            rnn_state_enc[i]:zero()
        end
        for t = source_l, 1, -1 do
            local encoder_input = { source_input[t], table.unpack(rnn_state_enc) }
            local out = model[4]:forward(encoder_input)
            rnn_state_enc = out
            context[{ {}, t }]:add(out[#out])
        end
        if model_opt.init_dec == 1 then
            for L = 1, model_opt.num_layers do
                rnn_state_dec[L * 2 - 1 + model_opt.input_feed]:add(rnn_state_enc[L * 2 - 1])
                rnn_state_dec[L * 2 + model_opt.input_feed]:add(rnn_state_enc[L * 2])
            end
        end
    end
    rnn_state_dec_gold = {}
    for i = 1, #rnn_state_dec do
        table.insert(rnn_state_dec_gold, rnn_state_dec[i][{ { 1 } }]:clone())
    end

    if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
        cutorch.setDevice(opt.gpuid2)
        context2:copy(context)
        context = context2
    end

    local gold_score = 0
    print("DECODER POS", saved_decoder_position)
    --SEB
    local position_tensor = torch.LongTensor({ saved_decoder_position })
    --initial write and then append
    if saved_decoder_position == 0 then
        decoderfile:write('offsets', position_tensor, offset_options)
    else
        decoderfile:append('offsets', position_tensor, offset_options)
    end

    rnn_state_dec = {}
    for i = 1, #init_fwd_dec do
        table.insert(rnn_state_dec, init_fwd_dec[i][{ { 1 } }]:zero())
    end
    if model_opt.init_dec == 1 then
        rnn_state_dec = rnn_state_dec_gold
    end
    local target_l = gold:size(1)
    for t = 2, target_l do
        local decoder_input1
        if model_opt.use_chars_dec == 1 then
            decoder_input1 = word2charidx_targ:index(1, gold[{ { t - 1 } }])
        else
            decoder_input1 = gold[{ { t - 1 } }]
        end
        local decoder_input
        if model_opt.attn == 1 then
            decoder_input = { decoder_input1, context[{ { 1 } }], table.unpack(rnn_state_dec) }
        else
            decoder_input = { decoder_input1, context[{ { 1 }, source_l }], table.unpack(rnn_state_dec) }
        end
        local out_decoder = model[2]:forward(decoder_input)
        local out = model[3]:forward(out_decoder[#out_decoder]) -- K x vocab_size
        rnn_state_dec = {} -- to be modified later
        if model_opt.input_feed == 1 then
            table.insert(rnn_state_dec, out_decoder[#out_decoder])
        end


        for j = 1, #out_decoder - 1 do
            table.insert(rnn_state_dec, out_decoder[j])
        end
        --save the decoder states and the attention
        if saved_decoder_position == 0 then
            if model_opt.attn == 1 then
                decoderfile:write("attention", decoder_softmax.output, state_options)
            end

            for k = 1, 2 * model_opt.num_layers do
                decoderfile:write("states" .. k, out_decoder[k], state_options)
            end
        else
            if model_opt.attn == 1 then
                decoderfile:append("attention", decoder_softmax.output, state_options)
            end

            for k = 1, 2 * model_opt.num_layers do
                decoderfile:append("states" .. k, out_decoder[k], state_options)
            end
        end
        saved_decoder_position = saved_decoder_position + 1

        gold_score = gold_score + out[1][gold[t]]
    end

    return max_score, gold_score
end

function idx2key(file)
    local f = io.open(file, 'r')
    local t = {}
    for line in f:lines() do
        local c = {}
        for w in line:gmatch '([^%s]+)' do
            table.insert(c, w)
        end
        t[tonumber(c[2])] = c[1]
    end
    return t
end

function flip_table(u)
    local t = {}
    for key, value in pairs(u) do
        t[value] = key
    end
    return t
end


function get_layer(layer)
    if layer.name ~= nil then
        if layer.name == 'decoder_attn' then
            decoder_attn = layer
        elseif layer.name:sub(1, 3) == 'hop' then
            hop_attn = layer
        elseif layer.name:sub(1, 7) == 'softmax' then
            table.insert(softmax_layers, layer)
        elseif layer.name == 'word_vecs_enc' then
            word_vecs_enc = layer
        elseif layer.name == 'word_vecs_dec' then
            word_vecs_dec = layer
        end
    end
end

function sent2wordidx(sent, word2idx, start_symbol)
    local t = {}
    local u = {}
    if start_symbol == 1 then
        table.insert(t, START)
        table.insert(u, START_WORD)
    end

    for word in sent:gmatch '([^%s]+)' do
        local idx = word2idx[word] or UNK
        table.insert(t, idx)
        table.insert(u, word)
    end
    if start_symbol == 1 then
        table.insert(t, END)
        table.insert(u, END_WORD)
    end
    return torch.LongTensor(t), u
end

function sent2charidx(sent, char2idx, max_word_l, start_symbol)
    local words = {}
    if start_symbol == 1 then
        table.insert(words, START_WORD)
    end
    for word in sent:gmatch '([^%s]+)' do
        table.insert(words, word)
    end
    if start_symbol == 1 then
        table.insert(words, END_WORD)
    end
    local chars = torch.ones(#words, max_word_l)
    for i = 1, #words do
        chars[i] = word2charidx(words[i], char2idx, max_word_l, chars[i])
    end
    return chars, words
end

function word2charidx(word, char2idx, max_word_l, t)
    t[1] = START
    local i = 2
    for _, char in utf8.next, word do
        char = utf8.char(char)
        local char_idx = char2idx[char] or UNK
        t[i] = char_idx
        i = i + 1
        if i >= max_word_l then
            t[i] = END
            break
        end
    end
    if i < max_word_l then
        t[i] = END
    end
    return t
end

function wordidx2sent(sent, idx2word, source_str, attn, skip_end)
    local t = {}
    local start_i, end_i
    skip_end = skip_start_end or true
    if skip_end then
        end_i = #sent - 1
    else
        end_i = #sent
    end
    for i = 2, end_i do -- skip START and END
    if sent[i] == UNK then
        if opt.replace_unk == 1 then
            local s = source_str[attn[i]]
            if phrase_table[s] ~= nil then
                print(s .. ':' .. phrase_table[s])
            end
            local r = phrase_table[s] or s
            table.insert(t, r)
        else
            table.insert(t, idx2word[sent[i]])
        end
    else
        table.insert(t, idx2word[sent[i]])
    end
    end
    return table.concat(t, ' ')
end

function clean_sent(sent)
    local s = stringx.replace(sent, UNK_WORD, '')
    s = stringx.replace(s, START_WORD, '')
    s = stringx.replace(s, END_WORD, '')
    s = stringx.replace(s, START_CHAR, '')
    s = stringx.replace(s, END_CHAR, '')
    return s
end

function strip(s)
    return s:gsub("^%s+", ""):gsub("%s+$", "")
end

function main()
    -- some globals
    PAD = 1; UNK = 2; START = 3; END = 4
    PAD_WORD = '<blank>'; UNK_WORD = '<unk>'; START_WORD = '<s>'; END_WORD = '</s>'
    START_CHAR = '{'; END_CHAR = '}'
    MAX_SENT_L = opt.max_sent_l
    assert(path.exists(opt.src_file), 'src_file does not exist')
    assert(path.exists(opt.model), 'model does not exist')

    -- parse input params
    opt = cmd:parse(arg)
    if opt.gpuid >= 0 then
        require 'cutorch'
        require 'cunn'
        if opt.cudnn == 1 then
            require 'cudnn'
        end
    end
    print('loading ' .. opt.model .. '...')
    checkpoint = torch.load(opt.model)
    print('done!')

    if opt.replace_unk == 1 then
        phrase_table = {}
        if path.exists(opt.srctarg_dict) then
            local f = io.open(opt.srctarg_dict, 'r')
            for line in f:lines() do
                local c = line:split("|||")
                phrase_table[strip(c[1])] = c[2]
            end
        end
    end


    -- load model and word2idx/idx2word dictionaries
    model, model_opt = checkpoint[1], checkpoint[2]
    for i = 1, #model do
        model[i]:evaluate()
    end
    -- for backward compatibility
    model_opt.brnn = model_opt.brnn or 0
    model_opt.input_feed = model_opt.input_feed or 1
    model_opt.attn = model_opt.attn or 1

    idx2word_src = idx2key(opt.src_dict)
    word2idx_src = flip_table(idx2word_src)
    idx2word_targ = idx2key(opt.targ_dict)
    word2idx_targ = flip_table(idx2word_targ)

    -- load character dictionaries if needed
    if model_opt.use_chars_enc == 1 or model_opt.use_chars_dec == 1 then
        utf8 = require 'lua-utf8'
        char2idx = flip_table(idx2key(opt.char_dict))
        model[1]:apply(get_layer)
    end
    if model_opt.use_chars_dec == 1 then
        word2charidx_targ = torch.LongTensor(#idx2word_targ, model_opt.max_word_l):fill(PAD)
        for i = 1, #idx2word_targ do
            word2charidx_targ[i] = word2charidx(idx2word_targ[i], char2idx,
                model_opt.max_word_l, word2charidx_targ[i])
        end
    end
    -- load gold labels if it exists
    if path.exists(opt.targ_file) then
        print('loading GOLD labels at ' .. opt.targ_file)
        gold = {}
        local file = io.open(opt.targ_file, 'r')
        for line in file:lines() do
            table.insert(gold, line)
        end
    else
        print('GOLD label file not found')
        os.exit()
    end

    if opt.gpuid >= 0 then
        cutorch.setDevice(opt.gpuid)
        for i = 1, #model do
            if opt.gpuid2 >= 0 then
                if i == 1 or i == 4 then
                    cutorch.setDevice(opt.gpuid)
                else
                    cutorch.setDevice(opt.gpuid2)
                end
            end
            model[i]:double():cuda()
            model[i]:evaluate()
        end
    end

    --options for saving
    offset_options = hdf5.DataSetOptions()
    offset_options:setChunked(1)

    state_options = hdf5.DataSetOptions()
    state_options:setChunked(1, 1)

    --ensure that file exists and open it in append mode
    encoderfile = hdf5.open("encoder.hdf5", "w")
    encoderfile:close()
    encoderfile = hdf5.open("encoder.hdf5", "r+")


    decoderfile = hdf5.open("decoder.hdf5", "w")
    decoderfile:close()
    decoderfile = hdf5.open("decoder.hdf5", "r+")

    saved_encoder_position = 0
    saved_decoder_position = 0

    softmax_layers = {}
    model[2]:apply(get_layer)
    if model_opt.attn == 1 then
        decoder_attn:apply(get_layer)
        decoder_softmax = softmax_layers[1]
    end


    context_proto = torch.zeros(1, MAX_SENT_L, model_opt.rnn_size)
    local h_init_dec = torch.zeros(1, model_opt.rnn_size)
    local h_init_enc = torch.zeros(1, model_opt.rnn_size)
    if opt.gpuid >= 0 then
        h_init_enc = h_init_enc:cuda()
        --        h_init_dec = h_init_dec:cuda()
        cutorch.setDevice(opt.gpuid)
        if opt.gpuid2 >= 0 then
            cutorch.setDevice(opt.gpuid)
            context_proto = context_proto:cuda()
            cutorch.setDevice(opt.gpuid2)
        else
            context_proto = context_proto:cuda()
        end
    end
    init_fwd_enc = {}
    init_fwd_dec = {} -- initial context
    if model_opt.input_feed == 1 then
        table.insert(init_fwd_dec, h_init_dec:clone())
    end

    for L = 1, model_opt.num_layers do
        table.insert(init_fwd_enc, h_init_enc:clone())
        table.insert(init_fwd_enc, h_init_enc:clone())
        table.insert(init_fwd_dec, h_init_dec:clone()) -- memory cell
        table.insert(init_fwd_dec, h_init_dec:clone()) -- hidden state
    end

    gold_score_total = 0
    gold_words_total = 0

    local sent_id = 0
    local file = io.open(opt.src_file, "r")

    for line in file:lines() do
        sent_id = sent_id + 1
        line = clean_sent(line)
        print('SENT ' .. sent_id .. ': ' .. line)
        local source, source_str
        if model_opt.use_chars_enc == 0 then
            source, source_str = sent2wordidx(line, word2idx_src, model_opt.start_symbol)
        else
            source, source_str = sent2charidx(line, char2idx, model_opt.max_word_l, model_opt.start_symbol)
        end
        target, target_str = sent2wordidx(gold[sent_id], word2idx_targ, 1)
        print("Sentence", sent_id, source:size(1))
        pred_score, gold_score = generate_beam(model, state, MAX_SENT_L, source, target)

        if gold ~= nil then
            print('GOLD ' .. sent_id .. ': ' .. gold[sent_id])
            print(string.format("GOLD SCORE: %.4f", gold_score))
            gold_score_total = gold_score_total + gold_score
            gold_words_total = gold_words_total + target:size(1) - 1
        end
    end
    print(string.format("GOLD AVG SCORE: %.4f, GOLD PPL: %.4f",
        gold_score_total / gold_words_total,
        math.exp(-gold_score_total / gold_words_total)))
end

main()

