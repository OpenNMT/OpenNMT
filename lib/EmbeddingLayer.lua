local constants = require 'lib.utils.constants'

local EmbeddingLayer, parent = torch.class('EmbeddingLayer', 'nn.Module')

function EmbeddingLayer:__init(vocab_size, vec_size, pre_trained, fix)
  parent.__init(self)

  self.net = nn.LookupTable(vocab_size, vec_size)

  -- If embeddings are given. Initialize them.
  if pre_trained:len() > 0 then
    local vecs = torch.load(pre_trained)
    self.net.weight:copy(vecs)
  end

  self.fix = fix

  -- Padding should not have any value.
  self.net.weight[constants.PAD]:zero()
end

function EmbeddingLayer:updateOutput(input)
  return self.net:forward(input)
end

function EmbeddingLayer:updateGradInput(input, gradOutput)
  self.gradInput = self.net:backward(input, gradOutput)

  if self.fix then
    -- Ignore gradients if embeddings are not to be optimized.
    self.net.gradWeight:zero()
  else
    -- Padding should not have any value.
    self.net.gradWeight[constants.PAD]:zero()
  end

  return self.gradInput
end

function EmbeddingLayer:parameters()
  return self.net:parameters()
end
