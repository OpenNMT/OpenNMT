--[[ Returns a tensor of the size of the input, filled with indexes over a given dimension.
Needed for positional embeddings.
--]]

local Position, parent = torch.class('onmt.Position', 'nn.Module')

--[[
Parameters:

  * `dimension` - the dimension to be indexed.
--]]

function Position:__init(dimension)
   parent.__init(self)
   self.dimension = dimension
end

function Position:updateOutput(input)
   local dim = self.dimension < 0 and input:dim() + self.dimension + 1 or self.dimension

   self.output = input

   for i=1,self.output:size(dim) do
     self.output:select(dim,i):fill(i)
   end

   return self.output
end

function Position:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput
   return self.gradInput
end
