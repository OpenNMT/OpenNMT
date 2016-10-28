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
  make_generator = make_generator
}
