local BatchTensor = torch.class('BatchTensor')

--[[
  Take Batch x TimeStep x layer size tensors
]]
function BatchTensor:__init(T)
  self.t = T
  self.sourceLength = T:size()[2]
  self.sourceSize = torch.LongTensor(T:size()[1]):fill(self.sourceLength)
  self.size = T:size()[1]
end

function BatchTensor:getSourceInput(t)
  return self.t:select(2,t)
end

return BatchTensor
