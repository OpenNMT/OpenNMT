--[[
  Input:
    - a sequence `seq` (batchxL)
    - a positional tensor (batch) with float position in [1;L]

  Output:
    - a local window [p-D;p+D] along second dimension around position p (float)
    - a gaussian distribution for the window $$\mu(s)=exp(-\frac{(s-p_t)^2}{2\sigma^2})$$
]]

local CenteredWindow, parent = torch.class('onmt.CenteredWindow', 'nn.Module')

function CenteredWindow:__init(D)
  parent.__init(self)
  self.D=D
  self.output = { torch.Tensor(), torch.Tensor() }
  --self.mask = torch.ByteTensor(1, 1)
  self.gradInput = { torch.Tensor(), torch.Tensor() }
  self.inv2sigma = 1/(D*D/2)
  self.dec = torch.Tensor(1, 2*D + 1)
  for j = 1, 2*self.D+1 do
    self.dec[1][j] = j-1-self.D
  end
end

function CenteredWindow:updateOutput(input)
  local seq = input[1]
  local p = input[2]

  self.output = { self.output[1]:typeAs(seq), self.output[2]:typeAs(seq) }

  local size = seq:size()
  local batch = size[1]
  local L = size[2]

  size[2] = 2*self.D + 1
  self.output[1]:resize(size):zero()
  --self.mask:resize(size):zero()

  local left = torch.floor(p) - self.D
  local right = torch.floor(p) + self.D

  for i = 1, batch do
    local dec = 1
    if left[i] < 1 then
      --self.mask[{i,{1,1-left}}]:fill(1)
      dec = 2 - left[i]
      left[i] = 1
    end
    if right[i] > L then
      --self.mask[{i,{2*self.D+1-(right-L-1), 2*self.D+1}}]:fill(1)
      right[i] = L
    end
    self.output[1][i]:narrow(1, dec, right[i]-left[i]+1):copy(seq[i]:narrow(1, left[i], right[i]-left[i]+1))
  end

  -- keep fractional part of p and expand to a batch * 2D+1 tensor
  local frac = (p - torch.floor(p)):resize(batch, 1):expand(batch, 2*self.D + 1)
  local decbatch = self.dec:expand(batch, 2*self.D + 1)
  self.output[2] = torch.exp(-self.inv2sigma*torch.cmul(frac+decbatch, frac+decbatch))

  return self.output
end

function CenteredWindow:updateGradInput(input, gradOutput)
  local seq = input[1]
  local p = input[2]

  -- passing back gradient through the window - ignore gradient propagation on mu
  self.gradInput[1] = self.gradInput[1]:typeAs(seq):resizeAs(seq):zero()

  local size = seq:size()
  local batch = size[1]
  local L = size[2]

  local left = torch.floor(p) - self.D
  local right = torch.floor(p) + self.D

  for i = 1, batch do
    local dec = 1
    if left[i] < 1 then
      dec = 2 - left[i]
      left[i] = 1
    end
    if right[i] > L then
      right[i] = L
    end
    self.gradInput[1][i]:narrow(1, left[i], right[i]-left[i]+1):copy(gradOutput[1][i]:narrow(1, dec, right[i]-left[i]+1))
  end

  -- differenciation on p
  local frac = (p - torch.floor(p)):resize(batch, 1):expand(batch, 2*self.D + 1)
  local decbatch = self.dec:expand(batch, 2*self.D + 1)
  self.gradInput[2] = -2*self.inv2sigma*torch.cmul(torch.cmul(self.output[2], frac+decbatch), gradOutput[2]):sum(2)

  return self.gradInput
end
