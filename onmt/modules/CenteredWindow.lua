--[[Build a local window [p-D;p+D] along second dimension around position p (float)]]

local CenteredWindow, parent = torch.class('onmt.CenteredWindow', 'nn.Module')

function CenteredWindow:__init(D)
   parent.__init(self)
   self.D=D
end

function CenteredWindow:updateOutput(input)
  self.output = self.output:typeAs(input[1])
  local size = input:size()
  size[2] = 2*self.D+1
  self.output:resize(size):zero()
  for i = 1, input[2]:size(1) do
    local left = math.floor(input[2][i] - self.D)
    local dec = 1
    if left < 1 then
      dec = 2-left
      left = 1
    end
    local right = math.floor(input[2][i] + self.D)
    if right > input[1]:size(2) then
      right = input[1]:size(2)
    end
    self.output[i]:narrow(1, dec, right-left):copy(input[1][i]:narrow(1, left, right-left))
  end
end
