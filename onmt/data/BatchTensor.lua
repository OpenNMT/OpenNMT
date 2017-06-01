local BatchTensor = torch.class('BatchTensor')

--[[
  Take Batch x TimeStep x layer size tensors
]]
function BatchTensor:__init(T, sizes)
  self.size = T:size()[1]
  self.sourceLength = T:size()[2]

  self.sourceSize = sizes or torch.LongTensor(self.size):fill(self.sourceLength)

  self.sourceInput = T
  self.sourceInputPadLeft = true

  self.sourceInputRev = self.sourceInput
    :index(2, torch.linspace(self.sourceLength, 1, self.sourceLength):long())
  self.sourceInputRevPadLeft = false
end

function BatchTensor:getSourceInput(t)
  return self.sourceInput:select(2, t)
end

function BatchTensor:variableLengths()
  return torch.any(torch.ne(self.sourceSize, self.sourceLength))
end

return BatchTensor
