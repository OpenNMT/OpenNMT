local model_utils = require 's2sa.utils.model_utils'
local table_utils = require 's2sa.utils.table_utils'
local cuda = require 's2sa.utils.cuda'
require 's2sa.sequencer'

local function masked_softmax(source_sizes, source_length, beam_size)
  local num_sents = #source_sizes
  local input = nn.Identity()()
  local softmax = nn.SoftMax()(input) -- beam_size*num_sents x State.source_length

  -- now we are masking the part of the output we don't need
  local tab
  if beam_size ~= nil then
    tab = nn.SplitTable(2)(nn.View(beam_size, num_sents, source_length)(softmax))
    -- num_sents x { beam_size x State.source_length }
  else
    tab = nn.SplitTable(1)(softmax) -- num_sents x { State.source_length }
  end

  local par = nn.ParallelTable()

  for b = 1, num_sents do
    local pad_length = source_length - source_sizes[b]
    local dim = 2
    if beam_size == nil then
      dim = 1
    end

    local seq = nn.Sequential()
    seq:add(nn.Narrow(dim, pad_length + 1, source_sizes[b]))
    seq:add(nn.Padding(1, -pad_length, 1, 0))
    par:add(seq)
  end

  local out_tab = par(tab) -- num_sents x { beam_size x State.source_length }
  local output = nn.JoinTable(1)(out_tab) -- num_sents*beam_size x State.source_length
  if beam_size ~= nil then
    output = nn.View(num_sents, beam_size, source_length)(output)
    output = nn.Transpose({1,2})(output) -- beam_size x num_sents x State.source_length
    output = nn.View(beam_size*num_sents, source_length)(output)
  else
    output = nn.View(num_sents, source_length)(output)
  end

  -- make sure the vector sums to 1 (softmax output)
  output = nn.Normalize(1)(output)

  return nn.gModule({input}, {output})
end


local Decoder, Sequencer = torch.class('Decoder', 'Sequencer')

function Decoder:__init(args, network)
  Sequencer.__init(self, 'dec', args, network)
  self.input_feed = args.input_feed
  self.mask_padding = args.mask_padding or false

  if self.mask_padding then
    -- get a reference on the attention layer to mask the attention softmax afterward
    self.network:apply(function (layer)
      if layer.name == 'decoder_attn' then
        self.decoder_attn = layer
      end
    end)
  end

  -- preallocate default input feeding tensor
  if self.input_feed then
    self.input_feed_proto = torch.zeros(args.max_batch_size, args.rnn_size)
  end

  if args.training then
    -- preallocate output gradients
    self.grad_out_proto = {}
    for _ = 1, args.num_layers do
      table.insert(self.grad_out_proto, torch.zeros(args.max_batch_size, args.rnn_size))
      table.insert(self.grad_out_proto, torch.zeros(args.max_batch_size, args.rnn_size))
    end
    table.insert(self.grad_out_proto, torch.zeros(args.max_batch_size, args.rnn_size))

    -- preallocate context gradient
    self.grad_context_proto = torch.zeros(args.max_batch_size, args.max_source_length, args.rnn_size)
  end
end

function Decoder:resize_proto(batch_size)
  Sequencer.resize_proto(self, batch_size)
  if self.input_feed then
    self.input_feed_proto:resize(batch_size, self.input_feed_proto:size(2))
  end
end

function Decoder:reset(source_sizes, source_length, beam_size)
  self.decoder_attn:replace(function(module)
    if module.name == 'softmax_attn' then
      local mod
      if source_sizes ~= nil then
        mod = masked_softmax(source_sizes, source_length, beam_size)
      else
        mod = nn.SoftMax()
      end

      mod.name = 'softmax_attn'
      mod = cuda.convert(mod)
      self.softmax_attn = mod
      return mod
    else
      return module
    end
  end)
end

function Decoder:forward_one(input, prev_states, context, prev_out, t)
  local inputs = {}
  table_utils.append(inputs, prev_states)
  table.insert(inputs, input)
  table.insert(inputs, context)
  if self.input_feed then
    if prev_out == nil then
      table.insert(inputs, self.input_feed_proto[{{1, input:size(1)}}])
    else
      table.insert(inputs, prev_out)
    end
  end

  if not self.eval_mode then
    -- remember inputs for the backward pass
    self.inputs[t] = inputs
  end

  local outputs = Sequencer.net(self, t):forward(inputs)

  local out = outputs[#outputs]
  local states = {}
  for i = 1, #outputs - 1 do
    table.insert(states, outputs[i])
  end

  return out, states
end

function Decoder:forward_and_apply(batch, encoder_states, context, func)
  local states = model_utils.copy_state(self.states_proto, encoder_states, batch.size)

  local prev_out

  for t = 1, batch.target_length do
    prev_out, states = self:forward_one(batch.target_input[t], states, context, prev_out, t)
    func(prev_out, t)
  end
end

function Decoder:forward(batch, encoder_states, context)
  if not self.eval_mode then
    self.inputs = {}
  end

  local outputs = {}

  self:forward_and_apply(batch, encoder_states, context, function (out)
    table.insert(outputs, out)
  end)

  return outputs
end

function Decoder:compute_score(batch, encoder_states, context, generator)
  local score = {}

  self:forward_and_apply(batch, encoder_states, context, function (out, t)
    local pred = generator.network:forward(out)
    for b = 1, batch.size do
      score[b] = score[b] or 0
      score[b] = score[b] + pred[b][batch.target_output[t][b]]
    end
  end)

  return score
end

function Decoder:compute_loss(batch, encoder_states, context, generator)
  local loss = 0

  self:forward_and_apply(batch, encoder_states, context, function (out, t)
    local pred = generator.network:forward(out)
    loss = loss + generator.criterion:forward(pred, batch.target_output[t])
  end)

  return loss
end

function Decoder:backward(batch, outputs, generator)
  local grad_states_input = model_utils.reset_state(self.grad_out_proto, batch.size)
  local grad_context_input = self.grad_context_proto[{{1, batch.size}, {1, batch.source_length}}]:zero()

  local grad_context_idx = #self.states_proto + 2
  local grad_input_feed_idx = #self.states_proto + 3

  local loss = 0

  for t = batch.target_length, 1, -1 do
    -- compute decoder output gradients
    local pred = generator.network:forward(outputs[t])
    loss = loss + generator.criterion:forward(pred, batch.target_output[t]) / batch.size
    local gen_grad_out = generator.criterion:backward(pred, batch.target_output[t])
    gen_grad_out:div(batch.size)
    local dec_grad_out = generator.network:backward(outputs[t], gen_grad_out)
    grad_states_input[#grad_states_input]:add(dec_grad_out)

    local grad_input = Sequencer.net(self, t):backward(self.inputs[t], grad_states_input)

    -- accumulate encoder output gradients
    grad_context_input:add(grad_input[grad_context_idx])

    grad_states_input[#grad_states_input]:zero()

    -- accumulate previous output gradients with input feeding gradients
    if self.input_feed and t > 1 then
      grad_states_input[#grad_states_input]:add(grad_input[grad_input_feed_idx])
    end

    -- prepare next decoder output gradients
    for i = 1, #self.states_proto do
      grad_states_input[i]:copy(grad_input[i])
    end
  end

  Sequencer.backward_word_vecs(self)

  return grad_states_input, grad_context_input, loss
end

function Decoder:convert(f)
  Sequencer.convert(self, f)
  self.input_feed_proto = f(self.input_feed_proto)

  if self.grad_context_proto ~= nil then
    self.grad_context_proto = f(self.grad_context_proto)
  end
end

return Decoder
