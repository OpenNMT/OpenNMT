require 'nngraph'

--[[ Sequencer is the base class for our time series LSTM models.
  Acts similarly to an `nn.Module`.
   Main task is to manage `self.network_clones`, the unrolled LSTM
  used during training.
  Classes encoder/decoder/biencoder generalize these definitions.
--]]
local Sequencer, parent = torch.class('onmt.Sequencer', 'nn.Module')

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
      input = onmt.EmbeddingLayer(args.vocab_size, input_size, args.pre_word_vecs, args.fix_word_vecs)(x)

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

    next_c, next_h = onmt.LSTM(input_size, args.rnn_size)({prev_c, prev_h, input}):split(2)

    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  -- For the decoder, compute the attention here using h^L as query.
  if model == 'dec' then
    local attn_layer = onmt.GlobalAttention(args.rnn_size)
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

--]]
function Sequencer:__init(model, args, network)
  parent.__init(self)

  self.network = network or build_network(model, args)
  self.network_clones = {}

  self.args = args

  -- Prototype for preallocated hidden and cell states.
  self.stateProto = torch.Tensor()

  -- Prototype for preallocated output gradients.
  self.gradOutputProto = torch.Tensor()
end

function Sequencer:_sharedClone()
  local params, gradParams
  if self.network.parameters then
    params, gradParams = self.network:parameters()
    if params == nil then
      params = {}
    end
  end

  local mem = torch.MemoryFile("w"):binary()
  mem:writeObject(self.network)

  -- We need to use a new reader for each clone.
  -- We don't want to use the pointers to already read objects.
  local reader = torch.MemoryFile(mem:storage(), "r"):binary()
  local clone = reader:readObject()
  reader:close()

  if self.network.parameters then
    local cloneParams, cloneGradParams = clone:parameters()
    for i = 1, #params do
      cloneParams[i]:set(params[i])
      cloneGradParams[i]:set(gradParams[i])
    end
  end

  mem:close()

  return clone
end

--[[Get a clone for a timestep.

Parameters:
  * `t` - timestep.

Returns: The raw network clone at timestep t.
  When `evaluate()` has been called, cheat and return t=1.
]]
function Sequencer:net(t)
  if self.train then
    -- In train mode, the network has to be cloned to remember intermediate
    -- outputs for each timestep and to allow backpropagation through time.
    if self.network_clones[t] == nil then
      local clone = self:_sharedClone()
      clone:training()
      self.network_clones[t] = clone
    end
    return self.network_clones[t]
  else
    if #self.network_clones > 0 then
      return self.network_clones[1]
    else
      return self.network
    end
  end
end

--[[ Tell the network to prepare for training mode. ]]
function Sequencer:training()
  parent.training(self)

  if #self.network_clones > 0 then
    -- Only first clone can be used for evaluation.
    self.network_clones[1]:training()
  end
end

--[[ Tell the network to prepare for evaluation mode. ]]
function Sequencer:evaluate()
  parent.evaluate(self)

  if #self.network_clones > 0 then
    self.network_clones[1]:evaluate()
  else
    self.network:evaluate()
  end
end

return Sequencer
