function nn.Module:reuseMem()
   self.reuse = true
   return self
end

function nn.Module:setReuse()
   if self.reuse then
      self.gradInput = self.output
   end
end

function make_lstm(data, opt, model, use_chars)
   assert(model == 'enc' or model == 'dec')
   local name = '_' .. model
   local dropout = opt.dropout or 0
   local n = opt.num_layers
   local rnn_size = opt.rnn_size
   local input_size
   if use_chars == 0 then
      input_size = opt.word_vec_size
   else
      input_size = opt.num_kernels
   end   
   local offset = 0
  -- there will be 2*n+3 inputs
   local inputs = {}
   table.insert(inputs, nn.Identity()()) -- x (batch_size x max_word_l)
   if model == 'dec' then
      table.insert(inputs, nn.Identity()()) -- all context (batch_size x source_l x rnn_size)
      table.insert(inputs, nn.Identity()()) -- prev context_attn (batch_size x rnn_size)
      offset = offset + 2
   end   
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
     -- c,h from previous timesteps
    local prev_c = inputs[L*2+offset]    
    local prev_h = inputs[L*2+1+offset]
    -- the input to this layer
    if L == 1 then
       if use_chars == 0 then
	  local word_vecs
	  if model == 'enc' then
	     word_vecs = nn.LookupTable(data.source_size, input_size)
	  else
	     word_vecs = nn.LookupTable(data.target_size, input_size)
	  end	  
	  word_vecs.name = 'word_vecs' .. name
	  x = word_vecs(inputs[1]) -- batch_size x word_vec_size
       else
	  local char_vecs = nn.LookupTable(data.char_size, opt.char_vec_size)
	  char_vecs.name = 'word_vecs' .. name
	  local charcnn = make_cnn(opt.char_vec_size,  opt.kernel_width, opt.num_kernels)
	  charcnn.name = 'charcnn' .. name
	  x = charcnn(char_vecs(inputs[1]))
	  if opt.num_highway_layers > 0 then
	     local mlp = make_highway(input_size, opt.num_highway_layers)
	     mlp.name = 'mlp' .. name
	     x = mlp(x)
	  end	  
       end
       input_size_L = input_size
       if model == 'dec' then
	  x = nn.JoinTable(2)({x, inputs[1+offset]}) -- batch_size x (word_vec_size + rnn_size)
	  input_size_L = input_size + rnn_size
       end
    else
       x = outputs[(L-1)*2]
       if opt.res_net == 1 and L > 2 then
	  x = nn.CAddTable()({x, outputs[(L-2)*2]})       
       end       
       input_size_L = rnn_size
       if opt.hop_attn == L and model == 'dec' then
	  local hop_attn = make_decoder_attn(data, opt, 1)
	  hop_attn.name = 'hop_attn' .. L
	  x = hop_attn({x, inputs[offset]})
       end
       if dropout > 0 then
	  x = nn.Dropout(dropout, nil, false)(x)
       end       
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size):reuseMem()(x)
    local h2h = nn.LinearNoBias(rnn_size, 4 * rnn_size):reuseMem()(prev_h)
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid():reuseMem()(n1)
    local forget_gate = nn.Sigmoid():reuseMem()(n2)
    local out_gate = nn.Sigmoid():reuseMem()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh():reuseMem()(n4)
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh():reuseMem()(next_c)})
    
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end
  if model == 'dec' then
     local top_h = outputs[#outputs]
     local decoder_attn = make_decoder_attn(data, opt)     
     decoder_attn.name = 'decoder_attn'
     local attn_out = decoder_attn({top_h, inputs[offset]})
     if dropout > 0 then
	attn_out = nn.Dropout(dropout, nil, false)(attn_out)
     end     
     table.insert(outputs, attn_out)
  end
  return nn.gModule(inputs, outputs)
end

function make_decoder_attn(data, opt, simple)
   -- 2D tensor target_t (batch_l x rnn_size) and
   -- 3D tensor for context (batch_l x source_l x rnn_size)

   local inputs = {}
   table.insert(inputs, nn.Identity()())
   table.insert(inputs, nn.Identity()())
   local target_t = nn.LinearNoBias(opt.rnn_size, opt.rnn_size)(inputs[1])
   local context = inputs[2]
   simple = simple or 0
   -- get attention

   local attn = nn.MM()({context, nn.Replicate(1,3)(target_t)}) -- batch_l x source_l x 1
   attn = nn.Sum(3)(attn)
   local softmax_attn = nn.SoftMax()
   softmax_attn.name = 'softmax_attn'
   attn = softmax_attn(attn)
   attn = nn.Replicate(1,2)(attn) -- batch_l x  1 x source_l
   
   -- apply attention to context
   local context_combined = nn.MM()({attn, context}) -- batch_l x 1 x rnn_size
   context_combined = nn.Sum(2)(context_combined) -- batch_l x rnn_size
   local context_output
   if simple == 0 then
      context_combined = nn.JoinTable(2)({context_combined, inputs[1]}) -- batch_l x rnn_size*2
      context_output = nn.Tanh()(nn.LinearNoBias(opt.rnn_size*2,
						 opt.rnn_size)(context_combined))
   else
      context_output = nn.CAddTable()({context_combined,inputs[1]})
   end   
   return nn.gModule(inputs, {context_output})   
end

function make_generator(data, opt)
   local model = nn.Sequential()
   model:add(nn.Linear(opt.rnn_size, data.target_size))
   model:add(nn.LogSoftMax())
   local w = torch.ones(data.target_size)
   w[1] = 0
   criterion = nn.ClassNLLCriterion(w)
   criterion.sizeAverage = false
   return model, criterion
end


-- cnn Unit
function make_cnn(input_size, kernel_width, num_kernels)
   local output
   local input = nn.Identity()() 
   if opt.cudnn == 1 then
      local conv = cudnn.SpatialConvolution(1, num_kernels, input_size,
					    kernel_width, 1, 1, 0)
      local conv_layer = conv(nn.View(1, -1, input_size):setNumInputDims(2)(input))
      output = nn.Sum(3)(nn.Max(3)(nn.Tanh()(conv_layer)))
   else
      local conv = nn.TemporalConvolution(input_size, num_kernels, kernel_width)
      local conv_layer = conv(input)
      output = nn.Max(2)(nn.Tanh()(conv_layer))
   end
   return nn.gModule({input}, {output})
end

function make_highway(input_size, num_layers, output_size, bias, f)
    -- size = dimensionality of inputs
    -- num_layers = number of hidden layers (default = 1)
    -- bias = bias for transform gate (default = -2)
    -- f = non-linearity (default = ReLU)
    
    local num_layers = num_layers or 1
    local input_size = input_size
    local output_size = output_size or input_size
    local bias = bias or -2
    local f = f or nn.ReLU()
    local start = nn.Identity()()
    local transform_gate, carry_gate, input, output
    for i = 1, num_layers do
       if i > 1 then
	  input_size = output_size
       else
	  input = start
       end       
       output = f(nn.Linear(input_size, output_size)(input))
       transform_gate = nn.Sigmoid()(nn.AddConstant(bias, true)(
					nn.Linear(input_size, output_size)(input)))
       carry_gate = nn.AddConstant(1, true)(nn.MulConstant(-1)(transform_gate))
       local proj
       if input_size==output_size then
	  proj = nn.Identity()
       else
	  proj = nn.LinearNoBias(input_size, output_size)
       end
       input = nn.CAddTable()({
	                     nn.CMulTable()({transform_gate, output}),
                             nn.CMulTable()({carry_gate, proj(input)})})
    end
    return nn.gModule({start},{input})
end

