require 'nn'
require 'nngraph'

local LSTM = require 's2sa.LSTM'

local function make_attention(opt)
  -- 2D tensor target_t (batch_l x rnn_size)
  -- 3D tensor context (batch_l x source_timesteps x rnn_size)

  local inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())
  local target_t = nn.Linear(opt.rnn_size, opt.rnn_size, false)(inputs[1])
  local context= inputs[2]
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
  context_output =  nn.Dropout(opt.dropout, nil, false)(context_output)

  return nn.gModule(inputs, {context_output})
end

local function make_lstm(vocab_size, opt, model)
  assert(model == 'enc' or model == 'dec')

  local inputs = {}
  for l = 1, opt.num_layers do
    table.insert(inputs, nn.Identity()()) -- h0: batch_size x rnn_size
  end
  table.insert(inputs, nn.Identity()()) -- x: batch_size x timesteps

  local hidden_states
  local outputs = {}

  for L = 1, opt.num_layers do
    local h0 = inputs[L]
    local x
    local input_size

    if L == 1 then
      input_size = opt.word_vec_size
      local word_vecs = nn.LookupTable(vocab_size, input_size)
      word_vecs.name = 'word_vecs' .. '_' .. model
      x = word_vecs(inputs[opt.num_layers + 1]) -- batch_size x timesteps x word_vec_size
    else
      input_size = opt.rnn_size
      x = nn.Dropout(opt.dropout, nil, false)(hidden_states) -- batch_size x timesteps x rnn_size
    end

    local lstm = nn.LSTM(input_size, opt.rnn_size)
    lstm.name = 'lstm'
    hidden_states = lstm({h0, x}) -- batch_size x timesteps x rnn_size

    local out = nn.Select(2, -1)(hidden_states) -- last hidden state: batch_size x rnn_size
    table.insert(outputs, out)
  end

  table.insert(outputs, hidden_states) -- a.k.a context for the encoder

  return nn.gModule(inputs, outputs)
end

local function make_generator(vocab_size, opt)
  local model = nn.Sequential()
  model:add(nn.Linear(opt.rnn_size, vocab_size))
  model:add(nn.LogSoftMax())
  local w = torch.ones(vocab_size)
  w[1] = 0
  local criterion = nn.ClassNLLCriterion(w)
  criterion.sizeAverage = false
  return criterion, model
end

return {
  make_attention = make_attention,
  make_lstm = make_lstm,
  make_generator = make_generator
}
