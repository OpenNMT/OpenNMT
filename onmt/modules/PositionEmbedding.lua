--[[ Returns a tensor of the size of the input, filled with indexes over a given dimension.
Needed for positional embeddings.
--]]

local PositionEmbedding, parent = torch.class('onmt.PositionEmbedding', 'nn.LookupTable')

--[[
Parameters:

  * `dimension` - the dimension to be indexed.
  * `nIndex` - maximum number of positions.
  * `nOutput` - positional embedding size.
--]]

function PositionEmbedding:__init(dimension, nIndex, nOutput)
   parent.__init(self, nIndex+1, nOutput)
   self.dimension = dimension
   self.max_pos = nIndex
   self.nOutput = nOutput
   self.input = torch.Tensor()
end

function PositionEmbedding:postParametersInitialization()
  self.weight[1]:zero()
end

function PositionEmbedding:updateOutput(input)

   local dim = self.dimension < 0 and input[1]:dim() + self.dimension + 1 or self.dimension

   self.input = self.input:typeAs(input[1])
   self.input:resizeAs(input[1][{{},{},1}])

   local sourceSize = input[2]

   for b=1,self.input:size(1) do
     local batch = self.input:select(1,b)
     local start = self.input:size(2) - sourceSize[b]
     for t=1,self.input:size(dim) do
       if t <= start then
         batch[t] = 1
       else
         batch[t] = math.min(t-start+1,self.max_pos+1)
       end
     end
   end

   self.output = parent.updateOutput(self, self.input)

   return self.output
end

function PositionEmbedding:updateGradInput(input, gradOutput)

   if torch.type(self.gradInput) ~= torch.type(input[1]) then
      self.gradInput = input[1].new()
   end
   if not self.gradInput:isSameSizeAs(input[1]) then
      self.gradInput:resizeAs(input[1]):zero()
   end

   return { self.gradInput, torch.zeros(input[2]:size()) }
end

function PositionEmbedding:accGradParameters(_, gradOutput, scale)
   parent.accGradParameters(self, self.input, gradOutput, scale)
   self.gradWeight[1]:zero()
end
