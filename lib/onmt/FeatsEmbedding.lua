local FeatsEmbedding, parent = torch.class('onmt.FeatsEmbedding', 'nn.Container')

function FeatsEmbedding:__init(dicts, dimExponent)
  parent.__init(self)

  self.net = self:_buildModel(dicts, dimExponent)
  self:add(self.net)
end

function FeatsEmbedding:_buildModel(dicts, dimExponent)
  local inputs = {}
  local output

  self.outputSize = 0
  self.embs = {}

  for i = 1, #dicts do
    local feat = nn.Identity()() -- batch_size
    table.insert(inputs, feat)

    local vocabSize = #dicts[i]
    local embSize = math.floor(vocabSize ^ dimExponent)

    self.embs[i] = onmt.WordEmbedding(vocabSize, embSize)
    local emb = self.embs[i](feat)

    self.outputSize = self.outputSize + embSize

    if not output then
      output = emb
    else
      output = nn.JoinTable(2)({output, emb})
    end
  end

  return nn.gModule(inputs, {output})
end

function FeatsEmbedding:updateOutput(input)
  self.output = self.net:updateOutput(input)
  return self.output
end

function FeatsEmbedding:updateGradInput(input, gradOutput)
  return self.net:updateGradInput(input, gradOutput)
end

function FeatsEmbedding:accGradParameters(input, gradOutput, scale)
  self.net:accGradParameters(input, gradOutput, scale)
end

function FeatsEmbedding:share(other, ...)
  for i = 1, #self.embs do
    self.embs[i]:share(other.embs[i], ...)
  end
end
