--[[ nn unit. Maps from word ids to embeddings. Slim wrapper around
nn.LookupTable to allow fixed and pretrained embeddings.
--]]
local WordEmbedding, parent = torch.class('onmt.WordEmbedding', 'onmt.Network')

--[[
Parameters:

  * `vocabSize` - size of the vocabulary
  * `vecSize` - size of the embedding
  * `preTrainined` - path to a pretrained vector file
  * `fix` - keep the weights of the embeddings fixed.
--]]
function WordEmbedding:__init(vocabSize, vecSize, preTrained, fix)
  self.vocabSize = vocabSize
  parent.__init(self, nn.LookupTable(vocabSize, vecSize, onmt.Constants.PAD))

  -- If embeddings are given. Initialize them.
  if preTrained and preTrained:len() > 0 then
    local vecs = torch.load(preTrained)
    self.net.weight:copy(vecs)

    self.fix = fix
    if self.fix then
      self.net.gradWeight = nil
    end
  end
end

function WordEmbedding:postParametersInitialization()
  self.net.weight[onmt.Constants.PAD]:zero()
end

function WordEmbedding:accGradParameters(input, gradOutput, scale)
  if not self.fix then
    self.net:accGradParameters(input, gradOutput, scale)
    self.net.gradWeight[onmt.Constants.PAD]:zero()
  end
end

function WordEmbedding:parameters()
  if not self.fix then
    return parent.parameters(self)
  end
end
