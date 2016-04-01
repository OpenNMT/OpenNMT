local Squeeze, parent = torch.class('nn.Squeeze', 'nn.Module')

function Squeeze:updateOutput(input)   
   self.output = input:squeeze()
   return self.output
end

function Squeeze:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput:view(input:size())
   return self.gradInput
end

local Unsqueeze, parent = torch.class('nn.Unsqueeze', 'nn.Module')

function Unsqueeze:updateOutput(input)
    self.output = input:view(1, input:size(1), -1)
    return self.output
end

function Unsqueeze:updateGradInput(input, gradOutput)
    self.gradInput = gradOutput:view(input:size())
    return self.gradInput
end

--LinearNoBias from elements library
local LinearNoBias, Linear = torch.class('nn.LinearNoBias', 'nn.Linear')

function LinearNoBias:__init(inputSize, outputSize)
   nn.Module.__init(self)

   self.weight = torch.Tensor(outputSize, inputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)

   self:reset()
end

function LinearNoBias:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
      end
   else
      self.weight:uniform(-stdv, stdv)
   end

   return self
end

function LinearNoBias:updateOutput(input)
   if input:dim() == 1 then
      self.output:resize(self.weight:size(1))
      self.output:mv(self.weight, input)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      self.output:resize(nframe, self.weight:size(1))
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      if not self.addBuffer or self.addBuffer:nElement() ~= nframe then
         self.addBuffer = input.new(nframe):fill(1)
      end
      self.output:addmm(0, self.output, 1, input, self.weight:t())
   else
      error('input must be vector or matrix')
   end

   return self.output
end

function LinearNoBias:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
      self.gradWeight:addr(scale, gradOutput, input)
   elseif input:dim() == 2 then
      self.gradWeight:addmm(scale, gradOutput:t(), input)
   end
end
