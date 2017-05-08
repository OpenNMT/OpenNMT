local JoinReplicateTable, parent = torch.class('onmt.JoinReplicateTable', 'nn.Module')

function JoinReplicateTable:__init(dimensionReplicate, dimensionJoin, nInputDims)
  parent.__init(self)
  self.size = torch.LongStorage()
  self.dimensionReplicate = dimensionReplicate
  self.dimensionJoin = dimensionJoin
  self.gradInput = {}
  self.nInputDims = nInputDims
end

function JoinReplicateTable:_getPositiveDimensions(input)
  local dimensionReplicate = self.dimensionReplicate
  local dimensionJoin = self.dimensionJoin
  if dimensionReplicate < 0 then
    dimensionReplicate = input[1]:dim() + dimensionReplicate + 1
  elseif self.nInputDims and input[1]:dim()==(self.nInputDims+1) then
    dimensionReplicate = dimensionReplicate + 1
  end
  if dimensionJoin < 0 then
    dimensionJoin = input[1]:dim() + dimensionJoin + 1
  elseif self.nInputDims and input[1]:dim()==(self.nInputDims+1) then
    dimensionJoin = dimensionJoin + 1
  end
  return dimensionReplicate, dimensionJoin
end

function JoinReplicateTable:_replicate(input, dim, dryRun)
  -- first replicate along dimensionReplicate, we assert all dimensions are the same but some can be one
  local maxSize = 0
  local hasOne = {}

  for i = 1, #input do
    local size = input[i]:size(dim)
    assert(maxSize == 0 or size == maxSize or size == 1 or maxSize == 1, "incorrect tensor size for replicate dimension")
    if size > maxSize then
      maxSize = size
    end
    if size == 1 then
      table.insert(hasOne, i)
    end
  end

  -- remember strides to restore after joining operation
  local memStrides = {}
  if maxSize > 1 and #hasOne > 0 then
    for i = 1, #hasOne do
      local idx = hasOne[i]
      local sz = input[idx]:size()
      sz[dim] = maxSize
      local st = input[idx]:stride()
      memStrides[idx] = st[dim]
      if not dryRun then
        st[dim] = 0
        input[idx]:set(input[idx]:storage(), input[idx]:storageOffset(), sz, st)
      end
    end
  end

  return memStrides
end

function JoinReplicateTable:_dereplicate(input, dim, memStrides)
  for idx, stval in ipairs(memStrides) do
    local sz = input[idx]:size()
    sz[dim] = 1
    local st = input[idx]:stride()
    st[dim] = stval
    input[idx]:set(input[idx]:storage(), input[idx]:storageOffset(), sz, st)
  end
end

function JoinReplicateTable:updateOutput(input)
  local dimensionReplicate, dimensionJoin = self:_getPositiveDimensions(input)

  local memStrides = self:_replicate(input, dimensionReplicate)

  for i=1,#input do
    local currentOutput = input[i]
    if i == 1 then
      self.size:resize(currentOutput:dim()):copy(currentOutput:size())
    else
      self.size[dimensionJoin] = self.size[dimensionJoin]
        + currentOutput:size(dimensionJoin)
    end
  end
  self.output:resize(self.size)

  local offset = 1
  for i=1,#input do
    local currentOutput = input[i]
    self.output:narrow(dimensionJoin, offset,
      currentOutput:size(dimensionJoin)):copy(currentOutput)
    offset = offset + currentOutput:size(dimensionJoin)
  end

  self:_dereplicate(input, dimensionReplicate, memStrides)

  return self.output
end

function JoinReplicateTable:updateGradInput(input, gradOutput)
  local dimensionReplicate, dimensionJoin = self:_getPositiveDimensions(input)

  local memStrides = self:_replicate(input, dimensionReplicate, true)

  for i=1,#input do
    if self.gradInput[i] == nil then
      self.gradInput[i] = input[i].new()
    end
    self.gradInput[i]:resizeAs(input[i])
  end

  -- clear out invalid gradInputs
  for i=#input+1, #self.gradInput do
    self.gradInput[i] = nil
  end

  local offset = 1
  for i=1,#input do
    local currentOutput = input[i]
    local currentGradInput = gradOutput:narrow(dimensionJoin, offset,
               currentOutput:size(dimensionJoin))
    if memStrides[i] then
      -- sum along the replicated dimension
      torch.sum(self.gradInput[i], currentGradInput, dimensionReplicate)
    else
      self.gradInput[i]:copy(currentGradInput)
    end

    offset = offset + currentOutput:size(dimensionJoin)
  end

  return self.gradInput
end

function JoinReplicateTable:type(type, tensorCache)
  self.gradInput = {}
  return parent.type(self, type, tensorCache)
end
