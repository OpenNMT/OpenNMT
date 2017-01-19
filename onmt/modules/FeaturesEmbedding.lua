--[[
  A nngraph unit that maps features ids to embeddings. When using multiple
  features this can be the concatenation or the sum of each individual embedding.
]]
local FeaturesEmbedding, parent = torch.class('onmt.FeaturesEmbedding', 'onmt.Network')

function FeaturesEmbedding:__init(dicts, dimExponent, dim, merge)
  parent.__init(self, self:_buildModel(dicts, dimExponent, dim, merge))
end

function FeaturesEmbedding:_buildModel(dicts, dimExponent, dim, merge)
  local inputs = {}
  local output

  if merge == 'sum' then
    self.outputSize = dim
  else
    self.outputSize = 0
  end

  for i = 1, #dicts do
    local feat = nn.Identity()() -- batchSize
    table.insert(inputs, feat)

    local vocabSize = dicts[i]:size()
    local embSize

    if merge == 'sum' then
      embSize = self.outputSize
    else
      embSize = math.floor(vocabSize ^ dimExponent)
      self.outputSize = self.outputSize + embSize
    end

    local emb = nn.LookupTable(vocabSize, embSize)(feat)

    if not output then
      output = emb
    elseif merge == 'sum' then
      output = nn.CAddTable()({output, emb})
    else
      output = nn.JoinTable(2)({output, emb})
    end
  end

  return nn.gModule(inputs, {output})
end
