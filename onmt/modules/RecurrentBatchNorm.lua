require('nngraph')

--[[
An implementation of recurrent batch normalization as explained in:
https://arxiv.org/abs/1603.09025

We compute per-time-step batch statistics as part of normalization. Each time step's
corresponding nn.BatchNormalization module is stored in self.modules, and the table
self.modules is *shared across all clones*.
--]]

local RecurrentBatchNorm, parent = torch.class('onmt.RecurrentBatchNorm', 'nn.Container')

--[[
Parameters:

  * `dim` - The dimension of the input
  * `eps` - Small number to add to avoid divide-by-zero (defaults to 1e-5)
  * `momentum` - Batch normalization momentum (defaults to 0.1)
  * `affine` - Whether to learn a scaling and bias term (see nn.BatchNormalization for details)
  * `noBias` - Whether to hold the bias to 0 (but learn the scaling term)
--]]
function RecurrentBatchNorm:__init(dim, eps, momentum, affine, noBias)
  parent.__init(self)

  -- store variables for later reconstruction
  self.dim = dim
  self.eps = eps
  self.momentum = momentum
  self.affine = affine
  self.noBias = noBias

  self:setTimeStep(1)
end

function RecurrentBatchNorm:__timeStepClone()
  local bn = nn.BatchNormalization(self.dim, self.eps, self.momentum, self.affine)
  bn:type(self._type)
  if self.affine then
    -- initialize weights using the recommended value
    bn.weight:fill(0.1)
    if self.noBias then
      bn.bias = nil
      bn.gradBias = nil
    else
      bn.bias:zero()
    end
  end
  return bn
end

--[[ Choose the correct nn.BatchNormalization to feed inputs through based on timestep ]]
function RecurrentBatchNorm:setTimeStep(t)
  self.t = t
  if self.train then
    while t > #self.modules do
      self:add(self:__timeStepClone())
    end
    self.model = self.modules[t]
  else
    self.model = self.modules[math.min(self.t, #self.modules)]
  end
end

function RecurrentBatchNorm:updateOutput(input)
  self.output = self.model:updateOutput(input)
  return self.output
end

function RecurrentBatchNorm:updateGradInput(input, gradOutput)
  return self.model:updateGradInput(input, gradOutput)
end

function RecurrentBatchNorm:accGradParameters(input, gradOutput, scale)
  return self.model:accGradParameters(input, gradOutput, scale)
end

--[[ Share the .modules table with another RecurrentBatchNorm module]]
function RecurrentBatchNorm:share(rbn, ...)
  local args = {...}
  for i,v in ipairs(args) do
    if v == 'modules' then
      self.modules = rbn.modules
      table.remove(args, i)
      break
    end
  end
  parent.share(self, rbn, table.unpack(args))
end
