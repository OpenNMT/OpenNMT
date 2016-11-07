require 'torch'
local cuda = require 's2sa.cuda'
local hdf5 = require 'hdf5'
local model_utils = require 's2sa.model_utils'

local function make_lstm(input_size, rnn_size)
  local inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())

  local prev_c = inputs[1]
  local prev_h = inputs[2]
  local x = inputs[3]

  -- evaluate the input sums at once for efficiency
  local i2h = nn.Linear(input_size, 4 * rnn_size)(x)
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
      nn.CMulTable()({in_gate, in_transform})
  })

  -- gated cells form the output
  local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

  return nn.gModule(inputs, {next_c, next_h})
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
  local softmax_attn = cuda.nn.SoftMax()
  softmax_attn.name = 'softmax_attn'
  attn = softmax_attn(attn)
  attn = nn.Replicate(1,2)(attn) -- batch_l x 1 x source_l

  -- apply attention to context
  local context_combined = nn.MM()({attn, context}) -- batch_l x 1 x rnn_size
  context_combined = nn.Sum(2)(context_combined) -- batch_l x rnn_size
  context_combined = nn.JoinTable(2)({context_combined, inputs[1]}) -- batch_l x rnn_size*2
  local context_output = nn.Tanh()(nn.Linear(opt.rnn_size*2,opt.rnn_size,false)(context_combined))

  return nn.gModule(inputs, {context_output})
end


local Sequencer = torch.class('Sequencer')

function Sequencer:__init(args, opt, model)
  self.network = cuda.convert(self:build_network(args.vocab_size, opt, model))
  self.network_clones = model_utils.clone_many_times(self.network, args.max_sent_length)

  if args.pre_word_vecs:len() > 0 then
    local f = hdf5.open(args.pre_word_vecs)
    local pre_word_vecs = f:read('word_vecs'):all()
    for i = 1, pre_word_vecs:size(1) do
      self.word_vecs.weight[i]:copy(pre_word_vecs[i])
    end
  end

  self.fix_word_vecs = args.fix_word_vecs
  self.word_vecs.weight[1]:zero()

  local h_init = cuda.convert(torch.zeros(opt.max_batch_size, opt.rnn_size))

  self.init_states = {}
  for _ = 1, opt.num_layers do
    table.insert(self.init_states, h_init:clone())
    table.insert(self.init_states, h_init:clone())
  end
end

function Sequencer:build_network(vocab_size, opt, model)
  local inputs = {}
  local outputs = {}

  local x
  local context
  local input_feed

  for _ = 1, opt.num_layers do
    table.insert(inputs, nn.Identity()()) -- c0: batch_size x rnn_size
    table.insert(inputs, nn.Identity()()) -- h0: batch_size x rnn_size
  end

  table.insert(inputs, nn.Identity()()) -- x: batch_size
  x = inputs[#inputs]

  if model == 'dec' then
    table.insert(inputs, nn.Identity()()) -- context: batch_size * source_length * rnn_size
    context = inputs[#inputs]
    if opt.input_feed then
      table.insert(inputs, nn.Identity()()) -- context: batch_size x rnn_size
      input_feed = inputs[#inputs]
    end
  end

  local next_c
  local next_h

  for L = 1, opt.num_layers do
    local input_size
    local input

    if L == 1 then
      input_size = opt.word_vec_size
      self.word_vecs = nn.LookupTable(vocab_size, input_size)
      input = self.word_vecs(x) -- batch_size x word_vec_size
      if model == 'dec' and opt.input_feed then
        input_size = input_size + opt.rnn_size
        input = nn.JoinTable(2)({input, input_feed})
      end
    else
      input_size = opt.rnn_size
      input = nn.Dropout(opt.dropout, nil, false)(next_h) -- batch_size x rnn_size
    end

    local prev_c = inputs[L*2 - 1]
    local prev_h = inputs[L*2]

    next_c, next_h = make_lstm(input_size, opt.rnn_size)({prev_c, prev_h, input}):split(2)

    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  if model == 'dec' then
    local attn_layer = make_attention(opt)
    attn_layer.name = 'decoder_attn'
    local attn_output = nn.Dropout(opt.dropout, nil, false)(attn_layer({next_h, context}))
    table.insert(outputs, attn_output)
  end

  return nn.gModule(inputs, outputs)
end

function Sequencer:backward_word_vecs()
  self.word_vecs.gradWeight[1]:zero()
  if self.fix_word_vecs == 1 then
    self.word_vecs.gradWeight:zero()
  end
end

function Sequencer:get_clone(t)
  if self.eval_mode then
    return self.network_clones[1]
  else
    return self.network_clones[t]
  end
end

function Sequencer:training()
  if self.eval_mode ~= nil and self.eval_mode then
    self.network_clones[1]:training()
  else
    for i = 1, #self.network_clones do
      self.network_clones[i]:training()
    end
  end

  self.eval_mode = false
end

function Sequencer:evaluate()
  self.network_clones[1]:evaluate()
  self.eval_mode = true
end

function Sequencer:float()
  self.network:float()
end

function Sequencer:double()
  self.network:double()
end

function Sequencer:cuda()
  self.network:cuda()
end


return Sequencer
