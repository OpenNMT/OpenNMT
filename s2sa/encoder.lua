require 'torch'
local hdf5 = require 'hdf5'
local model_utils = require 's2sa.model_utils'
local table_utils = require 's2sa.table_utils'

local Encoder = torch.class("Encoder")

function Encoder:__init(args)
  self.word_vecs_enc = args.word_vecs_enc
  self.fix_word_vecs_enc = args.fix_word_vecs_enc
  self.network = args.network

  if args.pre_word_vecs_enc:len() > 0 then
    local f = hdf5.open(args.pre_word_vecs_enc)
    local pre_word_vecs = f:read('word_vecs'):all()
    for i = 1, pre_word_vecs:size(1) do
      self.word_vecs_enc.weight[i]:copy(pre_word_vecs[i])
    end
  end

  self.word_vecs_enc.weight[1]:zero()
end

function Encoder:forget()
  self.network:apply(function (layer)
      if layer.name == 'lstm' then
        layer:resetStates()
      end
  end)
end

function Encoder:forward(inputs)
  local encoder_outputs = self.network:forward(inputs)

  local context = encoder_outputs[#encoder_outputs]
  table.remove(encoder_outputs)

  return encoder_outputs, context
end

function Encoder:backward(inputs, grad_output, fix_word)
  self.network:backward(inputs, grad_output)

  self.word_vecs_enc.gradWeight[1]:zero()
  if self.fix_word_vecs_enc == 1 then
    self.word_vecs_enc.gradWeight:zero()
  end
end

return Encoder
