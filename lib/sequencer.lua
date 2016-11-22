require 'torch'
local cuda = require 'lib.utils.cuda'
local constants = require 'lib.utils.constants'
local model_utils = require 'lib.utils.model_utils'
require 'lib.model'

--[[ Sequencer is the base class for our time series LSTM models.
  Acts similarly to an `nn.Module`.
   Main task is to manage `self.network_clones`, the unrolled LSTM
  used during training.
  Classes encoder/decoder/biencoder generalize these definitions.

  Inherits from [Model](lib+model).
--]]
local Sequencer, Model = torch.class('Sequencer', 'Model')

--[[Create a nngraph template.

Parameters:

  * `rnn_size` - internalsize
  * `input_size` - input size

Returns: An nngraph unit mapping: ${(c_{t-1}, h_{t-1}, x_t) => (c_{t}, h_{t})}$

--]]
local function make_lstm(input_size, rnn_size)

  local inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())

  local prev_c = inputs[1]
  local prev_h = inputs[2]
  local x = inputs[3]

  -- Evaluate the input sums at once for efficiency.
  local i2h = nn.Linear(input_size, 4 * rnn_size)(x)
  local h2h = nn.Linear(rnn_size, 4 * rnn_size, false)(prev_h)
  local all_input_sums = nn.CAddTable()({i2h, h2h})

  local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
  local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)

  -- Decode the gates.
  local in_gate = nn.Sigmoid()(n1)
  local forget_gate = nn.Sigmoid()(n2)
  local out_gate = nn.Sigmoid()(n3)

  -- Decode the write inputs.
  local in_transform = nn.Tanh()(n4)

  -- Perform the LSTM update.
  local next_c = nn.CAddTable()({
    nn.CMulTable()({forget_gate, prev_c}),
    nn.CMulTable()({in_gate, in_transform})
  })

  -- Gated cells form the output.
  local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

  local gmodule=nn.gModule(inputs, {next_c, next_h})
  gmodule.name='lstm'
  return gmodule
end

--[[ Create an nngraph attention unit of size `rnn_size`.

Returns: An nngraph unit mapping:
  ${(H_1 .. H_n, q) => (a)}$
  Where H is of `batch x n x rnn_size` and q is of `batch x rnn_size`.

  The full function is  ${\tanh(W_2 [(softmax((W_1 q + b_1) H) H), q] + b_2)}$.

TODO:

  * allow different query and context sizes.
  * don't use "rnn" (this function is more general)
--]]
local function make_attention(rnn_size)
  local inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())

  local target_t = nn.Linear(rnn_size, rnn_size, false)(inputs[1]) -- batch_l x rnn_size
  local context = inputs[2] -- batch_l x source_timesteps x rnn_size

  -- Get attention.
  local attn = nn.MM()({context, nn.Replicate(1,3)(target_t)}) -- batch_l x source_l x 1
  attn = nn.Sum(3)(attn)
  local softmax_attn = cuda.nn.SoftMax()
  softmax_attn.name = 'softmax_attn'
  attn = softmax_attn(attn)
  attn = nn.Replicate(1,2)(attn) -- batch_l x 1 x source_l

  -- Apply attention to context.
  local context_combined = nn.MM()({attn, context}) -- batch_l x 1 x rnn_size
  context_combined = nn.Sum(2)(context_combined) -- batch_l x rnn_size
  context_combined = nn.JoinTable(2)({context_combined, inputs[1]}) -- batch_l x rnn_size*2
  local context_output = nn.Tanh()(nn.Linear(rnn_size*2, rnn_size, false)(context_combined))

  return nn.gModule(inputs, {context_output})
end


--[[ Build one time-step of a stacked LSTM network

Parameters:

  * `model` - "dec" or "enc"
  * `args` - global args.

Returns: An nn-graph mapping

  $${(c^1_{t-1}, h^1_{t-1}, .., c^L_{t-1}, h^L_{t-1}, x_t, [con/H], [if]) =>
  (c^1_{t}, h^1_{t}, .., c^L_{t}, h^L_{t}, [a])}$$

  Where ${c^l}$ and ${h^l}$ are the hidden and cell states at each layer,
  ${x_t}$ is a sparse word to lookup,
  ${con/H}$ is the context/source hidden states for attention,
  ${if}$ is the input feeding, and
  ${a}$ is the context vector computed at this timestep.

TODO: remove lookup table from this function.
--]]
local function build_network(model, args)

  local inputs = {}
  local outputs = {}

  local x
  local context
  local input_feed

  -- Inputs are previous layers first and then x.
  for _ = 1, args.num_layers do
    table.insert(inputs, nn.Identity()()) -- c0: batch_size x rnn_size
    table.insert(inputs, nn.Identity()()) -- h0: batch_size x rnn_size
  end

  table.insert(inputs, nn.Identity()()) -- x: batch_size
  x = inputs[#inputs]

  -- Decoder needs the context (for attention) and optionally input feeding.
  if model == 'dec' then
    table.insert(inputs, nn.Identity()()) -- context: batch_size * source_length * rnn_size
    context = inputs[#inputs]
    if args.input_feed then
      table.insert(inputs, nn.Identity()()) -- input_feed: batch_size x rnn_size
      input_feed = inputs[#inputs]
    end
  end

  local next_c
  local next_h

  for L = 1, args.num_layers do
    local input_size
    local input

    if L == 1 then
      -- At first layer do word lookup.
      input_size = args.word_vec_size
      local word_vecs = nn.LookupTable(args.vocab_size, input_size)
      word_vecs.name = 'word_vecs'
      input = word_vecs(x) -- batch_size x word_vec_size

      -- If input feeding, concat previous to $x$.
      if model == 'dec' and args.input_feed then
        input_size = input_size + args.rnn_size
        input = nn.JoinTable(2)({input, input_feed})
      end
    else
      -- Otherwise just add dropout.
      input_size = args.rnn_size
      input = nn.Dropout(args.dropout, nil, false)(next_h) -- batch_size x rnn_size
    end

    local prev_c = inputs[L*2 - 1]
    local prev_h = inputs[L*2]

    next_c, next_h = make_lstm(input_size, args.rnn_size)({prev_c, prev_h, input}):split(2)

    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  -- For the decoder, compute the attention here using h^L as query.
  if model == 'dec' then
    local attn_layer = make_attention(args.rnn_size)
    attn_layer.name = 'decoder_attn'
    local attn_output = nn.Dropout(args.dropout, nil, false)(attn_layer({next_h, context}))
    table.insert(outputs, attn_output)
  end

  return nn.gModule(inputs, outputs)
end


--[[ Constructor

Parameters:

  * `model` - type of model (enc,dec)
  * `args` - global arguments
  * `network` - optional preconstructed network.

TODO: Should initialize all the members in this method.
   i.e. word_vecs, fix_word_vecs, network_clones, eval_mode, etc.

--]]
function Sequencer:__init(model, args, network)
  Model.__init(self)

  self.network = network or build_network(model, args)
  self.args = args

  -- Preallocate hidden states tensors for cell and hidden.
  self.states_proto = {}
  for _ = 1, args.num_layers do
    table.insert(self.states_proto, torch.zeros(args.max_batch_size, args.rnn_size))
    table.insert(self.states_proto, torch.zeros(args.max_batch_size, args.rnn_size))
  end
end

function Sequencer:resize_proto(batch_size)
  -- Call to change the `batch_size`.
  for i = 1, #self.states_proto do
    self.states_proto[i]:resize(batch_size, self.states_proto[i]:size(2))
  end
end

function Sequencer:backward_word_vecs()

  -- Padding should not have any value.
  self.word_vecs.gradWeight[constants.PAD]:zero()

  -- Case where the word vectors are given.
  if self.fix_word_vecs then
    self.word_vecs.gradWeight:zero()
  end
end

--[[Get a clone for a timestep.

Parameters:
  * `t` - timestep.

Returns: The raw network clone at timestep t.
  When `evaluate()` has been called, cheat and return t=1.
]]

function Sequencer:net(t)

  if self.network_clones == nil or t == nil then
    return self.network
  else
    if self.eval_mode then
      return self.network_clones[1]
    else
      return self.network_clones[t]
    end
  end
end

--[[ Tell the network to prepare for training mode. ]]
function Sequencer:training()
  if self.network_clones == nil then
    -- During training the model will clone itself `self.args.max_sent_length`
    -- times with shared parameters. This allows training to be done in a
    -- feed-forward style, with each input simply extending the network,
    -- and "backprop through time" consisting of `max_sent_length` steps.


    -- Clone network up to max_sent_length.
    self.network_clones = model_utils.clone_many_times(self.network, self.args.max_sent_length)
    for i = 1, #self.network_clones do
      self.network_clones[i]:training()
    end

    -- Lookup table.
    -- Get pointer to lookup table.
    self.network:apply(function (layer)
      if layer.name == 'word_vecs' then
        self.word_vecs = layer
      end
    end)

    -- If lookup table is given. Initialize it.
    if self.args.pre_word_vecs:len() > 0 then
      local vecs = torch.load(self.args.pre_word_vecs)
      self.word_vecs.weight:copy(vecs)
    end

    self.fix_word_vecs = self.args.fix_word_vecs
    self.word_vecs.weight[constants.PAD]:zero()
  else
    -- only first clone can be used for evaluation
    self.network_clones[1]:training()
  end

  self.eval_mode = false
end

--[[ Tell the network to prepare for evaluation mode. ]]
function Sequencer:evaluate()
  if self.network_clones == nil then
    self.network:evaluate()
  else
    self.network_clones[1]:evaluate()
  end

  self.eval_mode = true
end

function Sequencer:convert(f)

  f(self.network)

  for i = 1, #self.states_proto do
    self.states_proto[i] = f(self.states_proto[i])
  end

  if self.grad_out_proto ~= nil then
    for i = 1, #self.grad_out_proto do
      self.grad_out_proto[i] = f(self.grad_out_proto[i])
    end
  end
end

return Sequencer
