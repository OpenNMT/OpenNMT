--[[Build a local window [p-D;p+D] along second dimension around position p (float)]]

local CenteredWindow, parent = torch.class('onmt.CenteredWindow', 'nn.Module')

function CenteredWindow:__init(D)
  parent.__init(self)
  self.D=D
  self.gradInput = { torch.Tensor(), torch.Tensor() }
end

function CenteredWindow:updateOutput(input)
  self.output = self.output:typeAs(input[1])
  local size = input[1]:size()
  local L = size[2]
  size[2] = 2*self.D + 1
  self.output:resize(size):zero()
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
    self.output[i]:narrow(1, dec, right-left+1):copy(input[1][i]:narrow(1, left, right-left+1))
  end
  return self.output
end

function CenteredWindow:updateGradInput(input, gradOutput)
  -- passing back gradient around the window
  self.gradInput[1] = self.gradInput[1]:typeAs(gradOutput):resizeAs(input[1]):zero()
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
    self.gradInput[1][i]:narrow(1, left, right-left+1):copy(gradOutput[i]:narrow(1, dec, right-left+1))
  end

  -- no gradient on the positional tensor
  self.gradInput[2] = self.gradInput[2]:typeAs(gradOutput):resizeAs(input[2]):zero()

  return self.gradInput
end
