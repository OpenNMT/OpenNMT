local Linear = nn.Linear

--[[

  Extension of the Linear layer to allow the user to a set a collection of
  output row indices. When the row indices are set, the layer will behave like a
  Linear layer that output only these rows.
  This is particularily interesting for softmax reduction.

]]--

function Linear:RIndex_setOutputIndices(rowIndices)
  if self.rowIndices ~= rowIndices then
    if not self.fullWeight then
      self.fullWeight = self.weight:clone()
      self.fullBias = self.bias:clone()
    end
    if rowIndices then
      self.weight:resize(rowIndices:size(1), self.fullWeight:size(2))
                 :copy(self.fullWeight:index(1, rowIndices))
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

function Linear:RIndex_clean()
  if self.fullWeight then
    self.weight:resize(self.fullWeight:size()):copy(self.fullWeight)
    self.bias:resize(self.fullBias:size()):copy(self.fullBias)
  end
  self.fullWeight = nil
  self.fullBias = nil
  self.rowIndices = nil
end
