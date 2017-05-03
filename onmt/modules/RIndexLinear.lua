local RIndexLinear, parent = torch.class('onmt.RIndexLinear', 'nn.Linear')

--[[

  RIndexLinear is a Linear layer that allows the user to a set a collection of
  output row indices. When the row indices are set, the layer will behave like a
  Linear layer that output only these rows - all the other ones will be zero.
  This is particularily interesting for softmax reduction.

]]--

function RIndexLinear:__init(inputSize, outputSize, bias)
  parent.__init(self, inputSize, outputSize, bias)
  self.fullWeight = self.weight:clone()
  self.fullBias = self.bias:clone()
end

function RIndexLinear:setOutputIndices(rowIndices)
  if self.rowIndices ~= rowIndices then
    if rowIndices then
      self.weight:resize(rowIndices:size(1), self.fullWeight:size(2)):copy(self.fullWeight:index(1, rowIndices))
      self.bias:resize(rowIndices:size(1)):copy(self.fullBias:index(1, rowIndices))
    else
      self.fullWeight:indexCopy(1, self.rowIndices, self.weight)
      self.fullBias:indexCopy(1, self.rowIndices, self.bias)
      self.weight:resize(self.fullWeight:size()):copy(self.fullWeight)
      self.bias:resize(self.fullBias:size()):copy(self.fullBias)
    end
    self.gradWeight:resize(self.weight:size())
    self.gradBias:resize(self.bias:size())
    self.rowIndices = rowIndices
  end
end
