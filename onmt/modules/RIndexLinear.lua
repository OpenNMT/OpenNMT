local RIndexLinear, parent = torch.class('onmt.RIndexLinear', 'nn.Linear')

--[[

  MaskLinear is a Linear layer that allows the user to a set a collection of
  output row indices. When the row indices are set, the layer will behave like a
  Linear layer that output only these rows - all the other ones will be zero.
  This is particularily interesting for softmax reduction.

]]--

function RIndexLinear:__init(inputSize, outputSize, bias)
  parent.__init(self, inputSize, outputSize, bias)
  self.restrictedOutput = torch.Tensor(1)
end

function RIndexLinear:setOutputIndices(rowIndices)
  if self.rowIndices ~= rowIndices then
    self.rowIndices = rowIndices
    if rowIndices then
      self.restrictedWeight = self.weight:index(1, rowIndices)
      self.restrictedBias = self.bias:index(1, rowIndices)
    end
  end
end

local function updateAddBuffer(self, input)
   local nbatch = input:size(1)
   self.addBuffer = self.addBuffer or input.new()
   if self.addBuffer:nElement() ~= nbatch then
      self.addBuffer:resize(nbatch):fill(1)
   end
end

function RIndexLinear:updateOutput(input)
  if self.train or not self.rowIndices then
    return parent.updateOutput(self, input)
  end
  if input:dim() == 1 then
    self.output:resize(self.weight:size(1)):zero()
    self.restrictedOutput:resize(self.restrictedWeight:size(1))
    if self.bias then
      self.restrictedOutput:copy(self.restrictedBias)
    else
      self.restrictedOutput:zero()
    end
    self.restrictedOutput:addmv(1, self.restrictedWeight, input)
    self.output:indexCopy(1, self.rowIndices, self.restrictedOutput)
  elseif input:dim() == 2 then
    local nbatch = input:size(1)
    self.output:resize(nbatch, self.weight:size(1)):zero()
    updateAddBuffer(self, input)
    self.restrictedOutput:resize(nbatch, self.restrictedWeight:size(1)):zero()
    self.restrictedOutput:addmm(0, self.restrictedOutput, 1, input, self.restrictedWeight:t())
    if self.bias then
      self.restrictedOutput:addr(1, self.addBuffer, self.restrictedBias)
    end
    self.output:indexCopy(2, self.rowIndices, self.restrictedOutput)
  else
    error('input must be vector or matrix')
  end
  return self.output
end

function RIndexLinear:updateGradInput(input, gradOutput)
  if self.gradInput then
    if self.rowIndices then

      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
        self.gradInput:zero()
      end
      if input:dim() == 1 then
        local restrictedGradOutput = gradOutput:index(1, self.rowIndices)
        self.gradInput:addmv(0, 1, self.restrictedWeight:t(), restrictedGradOutput)
      elseif input:dim() == 2 then
        local restrictedGradOutput = gradOutput:index(2, self.rowIndices)
        self.gradInput:addmm(0, 1, restrictedGradOutput, self.restrictedWeight)
      end
      return self.gradInput
    else
      return parent.updateGradInput(self, input, gradOutput)
    end
  end
end

