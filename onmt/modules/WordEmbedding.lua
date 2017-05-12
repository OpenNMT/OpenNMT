--[[ nn unit. Maps from word ids to embeddings. Slim wrapper around
nn.LookupTable to allow fixed and pretrained embeddings.
--]]
local WordEmbedding, parent = torch.class('onmt.WordEmbedding', 'onmt.Network')

--[[
Parameters:

  * `vocabSize` - size of the vocabulary
  * `vecSize` - size of the embedding
  * `preTrained` - path to a pretrained vector file
  * `fix` - keep the weights of the embeddings fixed.
--]]
function WordEmbedding:__init(vocabSize, vecSize, preTrained, fix)
  parent.__init(self, nn.LookupTable(vocabSize, vecSize, onmt.Constants.PAD))

  self.preTrained = preTrained
  self.fix = fix
end

function WordEmbedding:postParametersInitialization()
  -- If embeddings are given. Initialize them.
  if self.preTrained and self.preTrained:len() > 0 then
    local vecs = torch.load(self.preTrained)
    assert(vecs:size(1)==self.net.weight:size(1), "pretrained embeddings should be exactly on vocabulary")
    assert(vecs:size(2)<= self.net.weight:size(2), "size of pretrained embeddings is larger than embedding size")
    self.net.weight:narrow(2,1,vec:size(2)):copy(vecs)
    self.preTrainedVecSize = vecs:size(2)
  end

  self.net.weight[onmt.Constants.PAD]:zero()
  self:fixEmbeddings(self.fix)
end

function WordEmbedding:fixEmbeddings(fix)
  if fix then
    if not(self.preTrainedVecSize) or self.preTrainedVecSize == self.net.weight:size(2) then
      self.net.gradWeight = nil
    end
  elseif not fix and not self.net.gradWeight then
    self.net.gradWeight = self.net.weight.new(self.net.weight:size()):zero()
  end
  self.fix = fix
end

function WordEmbedding:accGradParameters(input, gradOutput, scale)
  if self.net.gradWeight then
    self.net:accGradParameters(input, gradOutput, scale)
    self.net.gradWeight[onmt.Constants.PAD]:zero()
    if self.fix and self.preTrainedVecSize and self.preTrainedVecSize ~= self.net.weight:size(2) then
      -- partial fix on a pretrained - zero the gradient
      self.net.gradWeight:narrow(2,1,vec:size(2)):zero()
    end
  end
end

function WordEmbedding:parameters()
  if self.net.gradWeight then
    return parent.parameters(self)
  end
end
