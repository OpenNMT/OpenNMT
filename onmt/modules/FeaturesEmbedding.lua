--[[
  A nngraph unit that maps features ids to embeddings. When using multiple
  features this can be the concatenation or the sum of each individual embedding.
]]
local FeaturesEmbedding, parent = torch.class('onmt.FeaturesEmbedding', 'onmt.Network')

function FeaturesEmbedding:__init(vocabSizes, vecSizes, merge)
  assert(#vocabSizes == #vecSizes)

  if merge == 'sum' then
    for i = 2, #vecSizes do
      assert(vecSizes[i] == vecSizes[1], 'embeddings must have the same size when merging with a sum')
    end
    self.outputSize = vecSizes[1]
  else
    self.outputSize = 0
    for i = 1, #vecSizes do
      self.outputSize = self.outputSize + vecSizes[i]
    end
  end

  parent.__init(self, self:_buildModel(vocabSizes, vecSizes, merge))
end

function FeaturesEmbedding:_buildModel(vocabSizes, vecSizes, merge)
  local inputs = {}
  local outputs = {}

  for i = 1, #vocabSizes do
    table.insert(inputs, nn.Identity()())
    table.insert(outputs, nn.LookupTable(vocabSizes[i], vecSizes[i])(inputs[#inputs]))
  end

  local output

  if #outputs == 1 then
    output = outputs[1]
  else
    if merge == 'sum' then
      output = nn.CAddTable()(outputs)
    else
      output = nn.JoinTable(2, 2)(outputs)
    end
  end

  return nn.gModule(inputs, {output})
end
