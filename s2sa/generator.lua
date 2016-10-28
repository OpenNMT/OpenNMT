require 'torch'

local function build_attention(opt)
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

local function build_network(vocab_size, opt)
  print('build_network '..vocab_size)
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- decoder output
  table.insert(inputs, nn.Identity()()) -- context

  local out = build_attention(opt)({inputs[1], inputs[2]})
  local map = nn.Linear(opt.rnn_size, vocab_size)(out)
  local loglk = nn.LogSoftMax()(map)

  return nn.gModule(inputs, {loglk})
end


local Generator = torch.class('Generator')

function Generator:__init(args, opt)
  self.network = build_network(args.vocab_size, opt)
  self.num_layers = opt.num_layers
end

function Generator:training()
  self.network:training()
end

function Generator:evaluate()
  self.network:evaluate()
end

function Generator:build_criterion(vocab_size)
  local w = torch.ones(vocab_size)
  w[1] = 0
  self.criterion = nn.ClassNLLCriterion(w)
  self.criterion.sizeAverage = false
end

function Generator:process(batch, context, decoder_states, decoder_out)
  local grad_context = context:clone():zero()
  local decoder_grad_output = decoder_states
  for l = 1, self.num_layers do
    decoder_grad_output[l]:zero()
  end
  table.insert(decoder_grad_output, decoder_out:clone())

  local loss = 0

  for t = batch.target_length, 1, -1 do
    local out = decoder_out:select(2, t)

    local generator_output = self.network:forward({out, context})

    loss = loss + self.criterion:forward(generator_output, batch.target_output[{{}, t}]) / batch.size
    local criterion_grad_input = self.criterion:backward(generator_output, batch.target_output[{{}, t}]) / batch.size

    local generator_grad_input = self.network:backward({out, context}, criterion_grad_input)

    decoder_grad_output[#decoder_grad_output][{{}, t}]:copy(generator_grad_input[1])
    grad_context:add(generator_grad_input[2]) -- accumulate gradient of context
  end

  return decoder_grad_output, grad_context, loss
end


return Generator
