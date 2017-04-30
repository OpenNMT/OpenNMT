--[[ Wrapper around a single network. ]]
local Network, parent = torch.class('onmt.Network', 'nn.Container')

function Network:__init(net)
  parent.__init(self)
  self.net = net
  self:add(net)
end

function Network:set(net)
  self.modules = { net }
  self.net = net
end

function Network:updateOutput(input)
  self.output = self.net:updateOutput(input)
  return self.output
end

function Network:updateGradInput(input, gradOutput)
  self.gradInput = self.net:updateGradInput(input, gradOutput)
  return self.gradInput
end

function Network:accGradParameters(input, gradOutput, scale)
  self.net:accGradParameters(input, gradOutput, scale)
end
