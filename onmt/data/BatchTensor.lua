local BatchTensor = torch.class('BatchTensor')

--[[
  Take Batch x TimeStep x layer size tensors
]]
function BatchTensor:__init(T, sizes)
  self.size = T:size()[1]
  self.sourceLength = T:size()[2]
  self.sourceSize = sizes or torch.LongTensor(self.size):fill(self.sourceLength)
  self.sourceInput = T
end

function BatchTensor:getSourceInput(t)
  return self.sourceInput:select(2, t)
end

function BatchTensor:variableLengths()
  return onmt.data.Batch.variableLengths(self)
end

function BatchTensor:reverseSourceInPlace()
  if not self.sourceInputRev then
    self.sourceInputRev = self.sourceInput:clone()

    for b = 1, self.size do
      local reversedIndices = torch.linspace(self.sourceSize[b], 1, self.sourceSize[b]):long()
      local window = {b, {self.sourceLength - self.sourceSize[b] + 1, self.sourceLength}}
      self.sourceInputRev[window]:copy(self.sourceInput[window]:index(1, reversedIndices))
    end
  end

  self.sourceInput, self.sourceInputRev = self.sourceInputRev, self.sourceInput
end

return BatchTensor
