--[[
  Build:
    - a local window [p-D;p+D] along second dimension around position p (float)
    - a gaussian distribution for the window $$\mu(s)=exp(-\frac{(s-p_t)^2}{2\sigma^2})$$
]]

local CenteredWindow, parent = torch.class('onmt.CenteredWindow', 'nn.Module')

function CenteredWindow:__init(D)
  parent.__init(self)
  self.D=D
  self.output = { torch.Tensor(), torch.Tensor() }
  self.gradInput = { torch.Tensor(), torch.Tensor() }
  self.inv2sigma = 1/(D*D/2)
end

function CenteredWindow:updateOutput(input)
  self.output = { self.output[1]:typeAs(input[1]), self.output[2]:typeAs(input[1]) }
  local size = input[1]:size()
  local L = size[2]
  size[2] = 2*self.D + 1
  self.output[1]:resize(size):zero()
  self.output[2]:resize(size[1], size[2])
  for i = 1, input[2]:size(1) do
    local left = math.floor(input[2][i] - self.D)
    local dec = 1
    if left < 1 then
      dec = 2 - left
      left = 1
    end
    local right = math.floor(input[2][i] + self.D)
    if right > L then
      right = L
    end
    self.output[1][i]:narrow(1, dec, right-left+1):copy(input[1][i]:narrow(1, left, right-left+1))
    local frac = input[2][i] - math.floor(input[2][i])
    for j = 1, 2*self.D+1 do
      self.output[2][i][j] = math.exp(-self.inv2sigma*(frac+j-1-self.D)*(frac+j-1-self.D))
    end
  end
  return self.output
end

function CenteredWindow:updateGradInput(input, gradOutput)
  -- passing back gradient around the window - ignore gradient propagation on mu
  self.gradInput[1] = self.gradInput[1]:typeAs(gradOutput[1]):resizeAs(input[1]):zero()
  local size = input[1]:size()
  local L = size[2]
  for i = 1, input[2]:size(1) do
    local left = math.floor(input[2][i] - self.D)
    local dec = 1
    if left < 1 then
      dec = 2 - left
      left = 1
    end
    local right = math.floor(input[2][i] + self.D)
    if right > L then
      right = L
    end
    self.gradInput[1][i]:narrow(1, left, right-left+1):copy(gradOutput[1][i]:narrow(1, dec, right-left+1))
  end

  -- no gradient on the positional tensor
  self.gradInput[2] = self.gradInput[2]:typeAs(gradOutput[1]):resizeAs(input[2]):zero()

  return self.gradInput
end
