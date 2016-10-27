require 'torch'
local hdf5 = require 'hdf5'
local model_utils = require 's2sa.model_utils'

local Decoder = torch.class("Decoder")

function Decoder:__init(args)
  self.word_vecs_dec = args.word_vecs_dec
  self.fix_word_vecs_dec = args.fix_word_vecs_dec
  self.network = args.network

  if args.pre_word_vecs_dec:len() > 0 then
    local f = hdf5.open(args.pre_word_vecs_dec)
    local pre_word_vecs = f:read('word_vecs'):all()
    for i = 1, pre_word_vecs:size(1) do
      self.word_vecs_dec.weight[i]:copy(pre_word_vecs[i])
    end
  end

  self.word_vecs_dec.weight[1]:zero()
end

function Decoder:forget()
  self.network:apply(function (layer)
      if layer.name == 'lstm' then
        layer:resetStates()
      end
  end)
end

function Decoder:forward(inputs)
  local decoder_outputs = self.network:forward(inputs)
  return decoder_outputs[#decoder_outputs]
end

function Decoder:backward(inputs, grad_output, fix_word)
  local decoder_grad_input = self.network:backward(inputs, grad_output)

  self.word_vecs_dec.gradWeight[1]:zero()
  if self.fix_word_vecs_dec == 1 then
    self.word_vecs_dec.gradWeight:zero()
  end

  return decoder_grad_input
end

return Decoder
