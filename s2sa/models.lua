require 'nn'
require 'nngraph'

local function make_criterion(vocab_size)
  local w = torch.ones(vocab_size)
  w[1] = 0
  local criterion = nn.ClassNLLCriterion(w)
  criterion.sizeAverage = false
  return criterion
end

local function make_attention(opt)
  local inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())

  local target_t = nn.Linear(opt.rnn_size, opt.rnn_size, false)(inputs[1]) -- batch_l x rnn_size
  local context = inputs[2] -- batch_l x source_timesteps x rnn_size

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
  context_combined = nn.JoinTable(2)({context_combined, inputs[1]}) -- batch_l x rnn_size*2
  local context_output = nn.Tanh()(nn.Linear(opt.rnn_size*2,opt.rnn_size,false)(context_combined))
  context_output = nn.Dropout(opt.dropout, nil, false)(context_output)

  return nn.gModule(inputs, {context_output})
end

local function make_generator(vocab_size, opt)
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- decoder output
  table.insert(inputs, nn.Identity()()) -- context

  local out = make_attention(opt)({inputs[1], inputs[2]})
  local map = nn.Linear(opt.rnn_size, vocab_size)(out)
  local loglk = nn.LogSoftMax()(map)

  return nn.gModule(inputs, {loglk})
end

return {
  make_criterion = make_criterion,
  make_generator = make_generator
}
