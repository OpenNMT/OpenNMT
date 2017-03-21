

local Energy, parent = torch.class('onmt.Energy', 'nn.Module')


function Energy:__init(dim)
	
	--~ print(dim)
	parent.__init(self)
	self.weight = torch.Tensor(dim, 1)
		
	self.gradWeight = self.weight:clone()

	self.gradInput = torch.Tensor()
end


function Energy:updateOutput(input)

	assert(input:dim() == 3) -- expect 3D input
	
	local batchSize = input:size(1)
	local seqLength = input:size(2)
	
	
	self.output:resize(batchSize, seqLength, 1)
	
	
	for b = 1, batchSize do
		-- input[b] should have size: seqLength x dim
		-- self weight with size: dim x 1
		self.output[b]:mm(input[b], self.weight)
	end
	
	return self.output
	
end

function Energy:updateGradInput(input, gradOutput)
	
	local batchSize = input:size(1)
	local seqLength = input:size(2)
	
	self.gradInput:resizeAs(input) -- batchSize * seqLength * dim
	
	local transposed = self.weight:transpose(1, 2)
	
	for b = 1, batchSize do
		-- gradOutput[b] has size seqLength x 1
		-- weight has size dim x 1
		self.gradInput[b]:mm(gradOutput[b], transposed)
	end
	
	--~ self.weight:transpose(1, 2)
	
	return self.gradInput
end

function Energy:accGradParameters(input, gradOutput)

	local batchSize = input:size(1)
	local seqLength = input:size(2)
	
	for b = 1, batchSize do
		self.gradWeight:addmm(input[b]:transpose(1, 2), gradOutput[b])
	end
	
end

return Energy

