require 'nn'
require 'nngraph'

function make_lstm(data, opt, model)
  assert(model == 'enc' or model == 'dec')
  local name = '_' .. model
  local n = opt.num_layers
  local rnn_size = opt.rnn_size
  local input_size = opt.word_vec_size
  local offset = 0
  -- there will be 2*n+3 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x (batch_size x max_word_l)
  if model == 'dec' then
    table.insert(inputs, nn.Identity()()) -- all context (batch_size x source_l x rnn_size)
    offset = offset + 1
  end
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    local nameL=model..'_L'..L..'_'
    -- c,h from previous timesteps
    local prev_c = inputs[L*2+offset]
    local prev_h = inputs[L*2+1+offset]
    -- the input to this layer
    if L == 1 then
      local word_vecs
      if model == 'enc' then
        word_vecs = nn.LookupTable(data.source_size, input_size)
      else
        word_vecs = nn.LookupTable(data.target_size, input_size)
      end
      word_vecs.name = 'word_vecs' .. name
      x = word_vecs(inputs[1]) -- batch_size x word_vec_size
      input_size_L = input_size
    else
      x = outputs[(L-1)*2]
      input_size_L = rnn_size
      if opt.dropout > 0 then
        x = nn.Dropout(opt.dropout, nil, false)(x)
      end
    end

    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x)
    local h2h = nn.Linear(rnn_size, 4 * rnn_size, false)(prev_h)
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local out_gate = nn.Sigmoid()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)
    -- perform the LSTM update
    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate, in_transform})})
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  if model == 'dec' then
    local top_h = outputs[#outputs]
    local decoder_attn = make_decoder_attn(data, opt)
    decoder_attn.name = 'decoder_attn'
    local decoder_out = decoder_attn({top_h, inputs[2]})
    if opt.dropout > 0 then
      decoder_out = nn.Dropout(opt.dropout, nil, false)(decoder_out)
    end
    table.insert(outputs, decoder_out)
  end

  return nn.gModule(inputs, outputs)
end

function make_decoder_attn(data, opt)
  -- 2D tensor target_t (batch_l x rnn_size) and
  -- 3D tensor for context (batch_l x source_l x rnn_size)

  local inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())
  local target_t = nn.Linear(opt.rnn_size, opt.rnn_size, false)(inputs[1])
  local context = inputs[2]
  -- get attention

  local attn = nn.MM()({context, nn.Replicate(1,3)(target_t)}) -- batch_l x source_l x 1
  attn = nn.Sum(3)(attn)
  local softmax_attn = nn.SoftMax()
  softmax_attn.name = 'softmax_attn'
  attn = softmax_attn(attn)
  attn = nn.Replicate(1,2)(attn) -- batch_l x 1 x source_l

  -- apply attention to context
  local context_combined = nn.MM()({attn, context}) -- batch_l x 1 x rnn_size
  context_combined = nn.Sum(2)(context_combined) -- batch_l x rnn_size
  local context_output
  context_combined = nn.JoinTable(2)({context_combined, inputs[1]}) -- batch_l x rnn_size*2
  context_output = nn.Tanh()(nn.Linear(opt.rnn_size*2,opt.rnn_size,false)(context_combined))

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
