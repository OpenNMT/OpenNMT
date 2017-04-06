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
  parent.__init(self, nn.LookupTable(vocabSize, vecSize, onmt.Constants.PAD))

  self.preTrained = preTrained
  if fix then
    self.net.gradWeight = nil
  end
end

function WordEmbedding:postParametersInitialization()
  -- If embeddings are given. Initialize them.
  if self.preTrained and self.preTrained:len() > 0 then
    local vecs = torch.load(self.preTrained)
    self.net.weight:copy(vecs)
  end

  self.net.weight[onmt.Constants.PAD]:zero()
end

function WordEmbedding:fixEmbeddings(fix)
  if fix and self.net.gradWeight then
    self.net.gradWeight = nil
  elseif not fix and not self.net.gradWeight then
    self.net.gradWeight = self.net.weight.new(self.net.weight:size()):zero()
  end
end

function WordEmbedding:accGradParameters(input, gradOutput, scale)
  if self.net.gradWeight then
    self.net:accGradParameters(input, gradOutput, scale)
    self.net.gradWeight[onmt.Constants.PAD]:zero()
  end
end

function WordEmbedding:parameters()
  if self.net.gradWeight then
    return parent.parameters(self)
  end
end
