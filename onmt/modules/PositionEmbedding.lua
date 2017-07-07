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
   local dim = self.dimension < 0 and input:dim() + self.dimension + 1 or self.dimension

   self.input = self.input:typeAs(input)
   self.input:resizeAs(input):copy(input)

   for b=1,self.input:size(1) do
     local batch = self.input:select(1,b)
     local start = -1
     for t=1,self.input:size(dim) do
       if batch[t] == onmt.Constants.PAD then
         batch[t] = 1
         start = start+1
       else
         batch[t] = math.min(t-start,self.max_pos+1)
       end
     end
   end

   self.output = parent.updateOutput(self, self.input)

   return self.output
end

function PositionEmbedding:updateGradInput(_, gradOutput)
   self.gradInput = parent.updateGradInput(self, self.input, gradOutput)
   return self.gradInput
end

function PositionEmbedding:accGradParameters(_, gradOutput, scale)
   parent.accGradParameters(self, self.input, gradOutput, scale)
   self.gradWeight[self.max_pos+1]:zero()
end
