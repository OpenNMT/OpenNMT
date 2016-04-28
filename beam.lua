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
cmd:option('-src_file', '',[[Source sequence to decode (one line per sequence)]])
cmd:option('-targ_file', '', [[True target sequence (optional)]])
cmd:option('-output_file', 'pred.txt', [[Path to output the predictions (each line will be the
                                       decoded sequence]])
cmd:option('-src_dict', 'data/demo.src.dict', [[Path to source vocabulary (*.src.dict file)]])
cmd:option('-targ_dict', 'data/demo.targ.dict', [[Path to target vocabulary (*.targ.dict file)]])
cmd:option('-char_dict', 'data/demo.char.dict', [[If using chars, path to character 
                                                vocabulary (*.char.dict file)]])

-- beam search options
cmd:option('-beam', 5,[[Beam size]])
cmd:option('-max_sent_l', 250, [[Maximum sentence length. If any sequences in srcfile are longer
                               than this then it will error out]])
cmd:option('-simple', 0, [[If = 1, output prediction is simply the first time the top of the beam
                         ends with an end-of-sentence token. If = 0, the model considers all 
                         hypotheses that have been generated so far that ends with end-of-sentence 
                         token and takes the highest scoring of all of them.]])
cmd:option('-replace_unk', 0, [[Replace the generated UNK tokens with the source token that 
                              had the highest attention weight. If srctarg_dict is provided, 
                              it will lookup the identified source token and give the corresponding 
                              target token. If it is not provided (or the identified source token
                              does not exist in the table) then it will copy the source token]])
cmd:option('-srctarg_dict', 'data/en-de.dict', [[Path to source-target dictionary to replace UNK 
                             tokens. See README.md for the format this file should be in]])
cmd:option('-score_gold', 1, [[If = 1, score the log likelihood of the gold as well]])
cmd:option('-n_best', 1, [[If > 1, it will also output an n_best list of decoded sentences]])
cmd:option('-gpuid',  -1,[[ID of the GPU to use (-1 = use CPU)]])
cmd:option('-gpuid2', -1,[[Second GPU ID]])

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

local StateAll = torch.class("StateAll")

function StateAll.initial(start)
   return {start}
end

function StateAll.advance(state, token)
   local new_state = copy(state)
   table.insert(new_state, token)
   return new_state
end

function StateAll.disallow(out)
   local bad = {1, 3} -- 1 is PAD, 3 is BOS
   for j = 1, #bad do
      out[bad[j]] = -1e9
   end
end

function StateAll.same(state1, state2)
   for i = 2, #state1 do
      if state1[i] ~= state2[i] then
         return false
      end
   end
   return true
end

function StateAll.next(state)
   return state[#state]
end

function StateAll.heuristic(state)
   return 0
end

function StateAll.print(state)
   for i = 1, #state do
      io.write(state[i] .. " ")
   end
   print()
end


-- Convert a flat index to a row-column tuple.
function flat_to_rc(v, flat_index)
   local row = math.floor((flat_index - 1) / v:size(2)) + 1
   return row, (flat_index - 1) % v:size(2) + 1
end

function generate_beam(model, initial, K, max_sent_l, source, gold)
   --reset decoder initial states
   if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
      cutorch.setDevice(opt.gpuid)
   end
   local n = max_sent_l
  -- Backpointer table.
   local prev_ks = torch.LongTensor(n, K):fill(1)
   -- Current States.
   local next_ys = torch.LongTensor(n, K):fill(1)
   -- Current Scores.
   local scores = torch.FloatTensor(n, K)
   scores:zero()
   local source_l = math.min(source:size(1), opt.max_sent_l)
   local attn_argmax = {}   -- store attn weights
   attn_argmax[1] = {}

   local states = {} -- store predicted word idx
   states[1] = {}
   for k = 1, 1 do
      table.insert(states[1], initial)
      table.insert(attn_argmax[1], initial)
      next_ys[1][k] = State.next(initial)
   end

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
   local context = context_proto[{{}, {1,source_l}}]:clone() -- 1 x source_l x rnn_size
   
   for t = 1, source_l do
      local encoder_input = {source_input[t], table.unpack(rnn_state_enc)}
      local out = model[1]:forward(encoder_input)
      rnn_state_enc = out
      context[{{},t}]:copy(out[#out])
   end
   context = context:expand(K, source_l, model_opt.rnn_size)
   
   if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
      cutorch.setDevice(opt.gpuid2)
      local context2 = context_proto2[{{1, K}, {1, source_l}}]
      context2:copy(context)
      context = context2
   end

   rnn_state_dec = {}
   for i = 1, #init_fwd_dec do
      table.insert(rnn_state_dec, init_fwd_dec[i]:zero())
   end

   if model_opt.init_dec == 1 then
      for L = 1, model_opt.num_layers do
	 rnn_state_dec[L*2]:copy(rnn_state_enc[L*2-1]:expand(K, model_opt.rnn_size))
	 rnn_state_dec[L*2+1]:copy(rnn_state_enc[L*2]:expand(K, model_opt.rnn_size))
      end
   end
   out_float = torch.FloatTensor()
   
   local i = 1
   local done = false
   local max_score = -1e9
   local found_eos = false
   while (not done) and (i < n) do
      i = i+1
      states[i] = {}
      attn_argmax[i] = {}
      local decoder_input1
      if model_opt.use_chars_dec == 1 then
	 decoder_input1 = word2charidx_targ:index(1, next_ys:narrow(1,i-1,1):squeeze())
      else
	 decoder_input1 = next_ys:narrow(1,i-1,1):squeeze()
	 if opt.beam == 1 then
	    decoder_input1 = torch.LongTensor({decoder_input1})
	 end	
      end
      local decoder_input = {decoder_input1, context, table.unpack(rnn_state_dec)}
      local out_decoder = model[2]:forward(decoder_input)
      local out = model[3]:forward(out_decoder[#out_decoder]) -- K x vocab_size
      
      rnn_state_dec = {} -- to be modified later
      table.insert(rnn_state_dec, out_decoder[#out_decoder])
      for j = 1, #out_decoder - 1 do
	 table.insert(rnn_state_dec, out_decoder[j])
      end
      out_float:resize(out:size()):copy(out)
      for k = 1, K do
	 State.disallow(out_float:select(1, k))
	 out_float[k]:add(scores[i-1][k])
      end
      -- All the scores available.

       local flat_out = out_float:view(-1)
       if i == 2 then
          flat_out = out_float[1] -- all outputs same for first batch
       end

       if model_opt.start_symbol == 1 then
	  decoder_softmax.output[{{},1}]:zero()
	  decoder_softmax.output[{{},source_l}]:zero()
       end
       
       for k = 1, K do
          while true do
             local score, index = flat_out:max(1)
             local score = score[1]
             local prev_k, y_i = flat_to_rc(out_float, index[1])
             states[i][k] = State.advance(states[i-1][prev_k], y_i)
             local diff = true
             for k2 = 1, k-1 do
                if State.same(states[i][k2], states[i][k]) then
                   diff = false
                end
             end
	     
             if i < 2 or diff then		
		local max_attn, max_index = decoder_softmax.output[prev_k]:max(1)
		attn_argmax[i][k] = State.advance(attn_argmax[i-1][prev_k],max_index[1])
                prev_ks[i][k] = prev_k
                next_ys[i][k] = y_i
                scores[i][k] = score
                flat_out[index[1]] = -1e9
                break -- move on to next k 
             end
             flat_out[index[1]] = -1e9
          end
       end
       for j = 1, #rnn_state_dec do
	  rnn_state_dec[j]:copy(rnn_state_dec[j]:index(1, prev_ks[i]))
       end
       end_hyp = states[i][1]
       end_score = scores[i][1]
       end_attn_argmax = attn_argmax[i][1]
       if end_hyp[#end_hyp] == END then
	  done = true
	  found_eos = true
       else
	  for k = 1, K do
	     local possible_hyp = states[i][k]
	     if possible_hyp[#possible_hyp] == END then
		found_eos = true
		if scores[i][k] > max_score then
		   max_hyp = possible_hyp
		   max_score = scores[i][k]
		   max_attn_argmax = attn_argmax[i][k]
		end
	     end	     
	  end	  
       end       
   end
   local gold_score = 0
   if opt.score_gold == 1 then
      rnn_state_dec = {}
      for i = 1, #init_fwd_dec do
	 table.insert(rnn_state_dec, init_fwd_dec[i][{{1}}]:zero())
      end
      if model_opt.init_dec == 1 then
	 for L = 1, model_opt.num_layers do
	    rnn_state_dec[L*2]:copy(rnn_state_enc[L*2-1][{{1}}])
	    rnn_state_dec[L*2+1]:copy(rnn_state_enc[L*2][{{1}}])
	 end
      end
      local target_l = gold:size(1) 
      for t = 2, target_l do
	 local decoder_input1
	 if model_opt.use_chars_dec == 1 then
	    decoder_input1 = word2charidx_targ:index(1, gold[{{t-1}}])
	 else
	    decoder_input1 = gold[{{t-1}}]
	 end
	 local decoder_input = {decoder_input1, context[{{1}}], table.unpack(rnn_state_dec)}
	 local out_decoder = model[2]:forward(decoder_input)
	 local out = model[3]:forward(out_decoder[#out_decoder]) -- K x vocab_size
	 rnn_state_dec = {} -- to be modified later
	 table.insert(rnn_state_dec, out_decoder[#out_decoder])
	 for j = 1, #out_decoder - 1 do
	    table.insert(rnn_state_dec, out_decoder[j])
	 end
	 gold_score = gold_score + out[1][gold[t]]

      end      
   end
   if opt.simple == 1 or end_score > max_score or not found_eos then
      max_hyp = end_hyp
      max_score = end_score
      max_attn_argmax = end_attn_argmax
   end

   return max_hyp, max_score, max_attn_argmax, gold_score, states[i], scores[i], attn_argmax[i]
end

function idx2key(file)   
   local f = io.open(file,'r')
   local t = {}
   for line in f:lines() do
      local c = {}
      for w in line:gmatch'([^%s]+)' do
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
      elseif layer.name:sub(1,3) == 'hop' then
	 hop_attn = layer
      elseif layer.name:sub(1,7) == 'softmax' then
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
   
   for word in sent:gmatch'([^%s]+)' do
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
      table.insert(START_WORD)
   end   
   for word in sent:gmatch'([^%s]+)' do
      table.insert(words, word)
   end
   if start_symbol == 1 then
      table.insert(END_WORD)
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
      i = i+1
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
      end_i = #sent-1
   else
      end_i = #sent
   end   
   for i = 2, end_i do -- skip START and END
      if sent[i] == UNK then
	 if opt.replace_unk == 1 then
	    local s = source_str[attn[i]]
	    if phrase_table[s] ~= nil then
	       print(s .. ':' ..phrase_table[s])
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
   return s:gsub("^%s+",""):gsub("%s+$","")
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
   end      
   print('loading ' .. opt.model .. '...')
   checkpoint = torch.load(opt.model)
   print('done!')

   if opt.replace_unk == 1 then
      phrase_table = {}
      if path.exists(opt.srctarg_dict) then
	 local f = io.open(opt.srctarg_dict,'r')
	 for line in f:lines() do
	    local c = line:split("|||")
	    phrase_table[strip(c[1])] = c[2]
	 end
      end      
   end

   -- load model and word2idx/idx2word dictionaries
   model, model_opt = checkpoint[1], checkpoint[2]
   if model_opt.cudnn == 1 then
      require 'cudnn'
   end
   
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
      opt.score_gold = 0
   end

   if opt.gpuid >= 0 then
      cutorch.setDevice(opt.gpuid)
      for i = 1, #model do
	 if opt.gpuid2 >= 0 then
	    if i == 1 then
	       cutorch.setDevice(opt.gpuid)
	    else
	       cutorch.setDevice(opt.gpuid2)
	    end
	 end	 	       
	 model[i]:double():cuda()
	 model[i]:evaluate()
      end
   end

   softmax_layers = {}
   model[2]:apply(get_layer)
   decoder_attn:apply(get_layer)
   decoder_softmax = softmax_layers[1]
   attn_layer = torch.zeros(opt.beam, MAX_SENT_L)
   
   context_proto = torch.zeros(1, MAX_SENT_L, model_opt.rnn_size)
   local h_init_dec = torch.zeros(opt.beam, model_opt.rnn_size)
   local h_init_enc = torch.zeros(1, model_opt.rnn_size) 
   if opt.gpuid >= 0 then
      h_init_enc = h_init_enc:cuda()      
      h_init_dec = h_init_dec:cuda()
      cutorch.setDevice(opt.gpuid)
      if opt.gpuid2 >= 0 then
	 cutorch.setDevice(opt.gpuid)
	 context_proto = context_proto:cuda()	 
	 cutorch.setDevice(opt.gpuid2)
	 context_proto2 = torch.zeros(opt.beam, MAX_SENT_L, model_opt.rnn_size):cuda()
      else
	 context_proto = context_proto:cuda()
      end
      attn_layer = attn_layer:cuda()
   end
   init_fwd_enc = {}
   init_fwd_dec = {h_init_dec:clone()} -- initial context   
   for L = 1, model_opt.num_layers do
      table.insert(init_fwd_enc, h_init_enc:clone())
      table.insert(init_fwd_enc, h_init_enc:clone())
      table.insert(init_fwd_dec, h_init_dec:clone()) -- memory cell
      table.insert(init_fwd_dec, h_init_dec:clone()) -- hidden state      
   end      
     
   pred_score_total = 0
   gold_score_total = 0
   pred_words_total = 0
   gold_words_total = 0
   
   State = StateAll
   local sent_id = 0
   pred_sents = {}
   local file = io.open(opt.src_file, "r")
   local out_file = io.open(opt.output_file,'w')   
   for line in file:lines() do
      sent_id = sent_id + 1
      line = clean_sent(line)      
      print('SENT ' .. sent_id .. ': ' ..line)
      local source, source_str
      if model_opt.use_chars_enc == 0 then
	 source, source_str = sent2wordidx(line, word2idx_src, model_opt.start_symbol)
      else
	 source, source_str = sent2charidx(line, char2idx, model_opt.max_word_l, model_opt.start_symbol)
      end
      if opt.score_gold == 1 then
	 target, target_str = sent2wordidx(gold[sent_id], word2idx_targ, 1)
      end
      state = State.initial(START)
      pred, pred_score, attn, gold_score, all_sents, all_scores, all_attn = generate_beam(model,
  		state, opt.beam, MAX_SENT_L, source, target)
      pred_score_total = pred_score_total + pred_score
      pred_words_total = pred_words_total + #pred - 1
      pred_sent = wordidx2sent(pred, idx2word_targ, source_str, attn, true)
      out_file:write(pred_sent .. '\n')      
      print('PRED ' .. sent_id .. ': ' .. pred_sent)
      if gold ~= nil then
	 print('GOLD ' .. sent_id .. ': ' .. gold[sent_id])
	 if opt.score_gold == 1 then
	    print(string.format("PRED SCORE: %.4f, GOLD SCORE: %.4f", pred_score, gold_score))
	    gold_score_total = gold_score_total + gold_score
	    gold_words_total = gold_words_total + target:size(1) - 1	 	    
	 end
      end
      if opt.n_best > 1 then
	 for n = 1, opt.n_best do
	    pred_sent_n = wordidx2sent(all_sents[n], idx2word_targ, source_str, all_attn[n], false)
	    local out_n = string.format("%d ||| %s ||| %.4f", n, pred_sent_n, all_scores[n])
	    print(out_n)
	    out_file:write(out_n .. '\n')
	 end	 
      end
      
      print('')
   end
   print(string.format("PRED AVG SCORE: %.4f, PRED PPL: %.4f", pred_score_total / pred_words_total,
		       math.exp(-pred_score_total/pred_words_total)))
   if opt.score_gold == 1 then      
      print(string.format("GOLD AVG SCORE: %.4f, GOLD PPL: %.4f",
			  gold_score_total / gold_words_total,
			  math.exp(-gold_score_total/gold_words_total)))
   end
   out_file:close()
end
main()

