local JoinReplicateTable, parent = torch.class('onmt.JoinReplicateTable', 'nn.Module')

function JoinReplicateTable:__init(dimensionReplicate, dimensionJoin, nInputDims)
   parent.__init(self)
   self.size = torch.LongStorage()
   self.dimension_replicate = dimensionReplicate
   self.dimension_join = dimensionJoin
   self.gradInput = {}
   self.nInputDims = nInputDims
end

function JoinReplicateTable:_getPositiveDimensions(input)
   local dimension_replicate = self.dimension_replicate
   local dimension_join = self.dimension_join
   if dimension_replicate < 0 then
      dimension_replicate = input[1]:dim() + dimension_replicate + 1
   elseif self.nInputDims and input[1]:dim()==(self.nInputDims+1) then
      dimension_replicate = dimension_replicate + 1
   end
   if dimension_join < 0 then
      dimension_join = input[1]:dim() + dimension_join + 1
   elseif self.nInputDims and input[1]:dim()==(self.nInputDims+1) then
      dimension_join = dimension_join + 1
   end
   return dimension_replicate, dimension_join
end

function JoinReplicateTable:_replicate(input, dim, dry_run)
   -- first replicate along dimension_replicate, we assert all dimensions are the same but some can be one
   local max_size = 0
   local has_one = {}

   for i = 1, #input do
      local size = input[i]:size(dim)
      assert(max_size == 0 or size == max_size or size == 1 or max_size == 1, "incorrect tensor size for replicate dimension")
      if size > max_size then
         max_size = size
      end
      if size == 1 then
         table.insert(has_one, i)
      end
   end

   -- remember strides to restore after joining operation
   local memStrides = {}
   if max_size > 1 and #has_one > 0 then
      for i = 1, #has_one do
         local idx = has_one[i]
         local sz = input[idx]:size()
         sz[dim] = max_size
         local st = input[idx]:stride()
         memStrides[idx] = st[dim]
         if not dry_run then
            st[dim] = 0
            input[idx] = torch.Tensor(input[idx]:storage(), input[idx]:storageOffset(), sz, st)
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
      input[idx] = torch.Tensor(input[idx]:storage(), input[idx]:storageOffset(), sz, st)
   end
end

function JoinReplicateTable:updateOutput(input)
   local dimension_replicate, dimension_join = self:_getPositiveDimensions(input)

   local memStrides = self:_replicate(input, dimension_replicate)

   for i=1,#input do
      local currentOutput = input[i]
      if i == 1 then
         self.size:resize(currentOutput:dim()):copy(currentOutput:size())
      else
         self.size[dimension_join] = self.size[dimension_join]
            + currentOutput:size(dimension_join)
      end
   end
   self.output:resize(self.size)

   local offset = 1
   for i=1,#input do
      local currentOutput = input[i]
      self.output:narrow(dimension_join, offset,
         currentOutput:size(dimension_join)):copy(currentOutput)
      offset = offset + currentOutput:size(dimension_join)
   end

   self:_dereplicate(input, dimension_replicate, memStrides)

   return self.output
end

function JoinReplicateTable:updateGradInput(input, gradOutput)
   local dimension_replicate, dimension_join = self:_getPositiveDimensions(input)

   local memStrides = self:_replicate(input, dimension_replicate, true)

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
      local currentGradInput = gradOutput:narrow(dimension_join, offset,
                      currentOutput:size(dimension_join))
      if memStrides[i] then
         -- sum along the replicated dimension
         self.gradInput[i]:copy(currentGradInput:sum(dimension_replicate))
      else
         self.gradInput[i]:copy(currentGradInput)
      end

      offset = offset + currentOutput:size(dimension_join)
   end

   return self.gradInput
end

function JoinReplicateTable:type(type, tensorCache)
   self.gradInput = {}
   return parent.type(self, type, tensorCache)
end
