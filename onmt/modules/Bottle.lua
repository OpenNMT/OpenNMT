-- Bottle extension to support tables from https://github.com/torch/nn/pull/1183
-- added also https://github.com/torch/nn/pull/1198

local Bottle, parent = torch.class("onmt.Bottle", "nn.Container")
local unpack = unpack or table.unpack

function Bottle:__init(module, nInputDim, nOutputDim)
  parent.__init(self)
  self.nInputDim = nInputDim or 2
  self.nOutputDim = nOutputDim or self.nInputDim
  self.dimDelta = self.nInputDim - self.nOutputDim
  -- Used to reshape the gradients
  self.inShape = torch.Tensor(self.nInputDim)
  self.outShape = torch.Tensor(self.nOutputDim)
  -- add module to modules
  self.modules[1] = module
end

function Bottle:dim(input)
  if torch.type(input) == 'table' then
    return input[1]:dim()
  else
    return input:dim()
  end
end

function Bottle:size(input)
  if torch.type(input) == 'table' then
    local size = {}
    for i=1,#input do
      table.insert(size, self:size(input[i]))
    end
    return size
  else
    return torch.LongTensor(input:size())
   end
end

function Bottle:bottle(input, batchDims)
  if torch.type(input) == 'table' then
    local inShape = {}
    local inSize, squeezeSize = nil, nil
    for i=1,#input do
      shape, inSize, squeezeSize = self:bottle(input[i], batchDims)
      table.insert(inShape, shape)
    end
    return inShape, inSize, squeezeSize
  else
    local inSize = torch.LongTensor(input:size())
    local squeezeSize = inSize[{{1, batchDims - 1}}]:prod()
    local inShape = inSize[{{batchDims, input:dim()}}]:clone()
    inShape[{{1}}]:mul(squeezeSize)
    -- Forward with the module's dimension
    return inShape, inSize, squeezeSize
  end
end

function Bottle:squeeze(input, inShape)
  if torch.type(input) == 'table' then
     input_ = {}
     for i=1,#input do
       input_[i] = self:squeeze(input[i], inShape[i])
     end
     return input_
  else
     return input:view(unpack(inShape:totable()))
  end
end

function Bottle:unbottle(output, batchDims, inSize, squeezeSize)
  if torch.type(output) == 'table' then
    local outShape = {}
    for i=1,#output do
      local output_, outShape_ = self:unbottle(output[i], batchDims, inSize, squeezeSize)
      table.insert(outShape, outShape_)
      output[i] = output_
    end
    return output, outShape
  else
    local outShape = torch.LongTensor(output:size())
    if math.abs(self.dimDelta) > 0 then
      inSize = inSize[{{1, inSize:size(1) - self.dimDelta}}]:clone()
    end
    inSize[{{batchDims, inSize:size(1)}}]:copy(outShape)
    inSize[{{batchDims}}]:div(squeezeSize)
    output = output:view(unpack(torch.totable(inSize)))
    return output, outShape
  end
end

function Bottle:updateOutput(input)
  -- first batchDims dimensions will be fused
  local batchDims = self:dim(input) - self.nInputDim + 1
  -- see if bottle is required
  if batchDims > 1 then
    -- bottle the first dims
    local inShape, inSize, squeezeSize = self:bottle(input, batchDims)
    self.inShape = inShape
    local newInput = self:squeeze(input, inShape)
    -- Forward with the module's dimension
    local output = self.modules[1]:updateOutput(newInput)
    assert(self:dim(output) == self.nOutputDim,
      "Wrong number of output dims on module. Expected: " ..
      self.nOutputDim .. ' but got ' ..
      tostring(output and self:dim(output)))

    -- unbottle
    self.output, self.outShape = self:unbottle(output, batchDims, inSize, squeezeSize)
  else
    self.output = self.modules[1]:updateOutput(input)
  end
  return self.output
end

function Bottle:updateGradInput(input, gradOutput)
  if self:dim(input) > self.nInputDim then
    local input_ = self:squeeze(input, self.inShape)
    local gradOutput_ = self:squeeze(gradOutput, self.outShape)
    self.modules[1]:updateGradInput(input_, gradOutput_)
    if self.modules[1].gradInput then
      self.gradInput = self:squeeze(self.modules[1].gradInput, self:size(input))
    else
      self.gradInput = nil
    end
  else
    if self.modules[1].gradInput then
      self.gradInput = self.modules[1]:updateGradInput(input, gradOutput)
    else
      self.gradInput = nil
    end
  end
  return self.gradInput
end

function Bottle:accGradParameters(input, gradOutput, scale)
  if self:dim(input) > self.nInputDim then
    input = self:squeeze(input, self.inShape)
    gradOutput = self:squeeze(gradOutput, self.outShape)
  end
  self.modules[1]:accGradParameters(input, gradOutput, scale)
end
