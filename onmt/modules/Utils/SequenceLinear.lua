-- Applying a linear layer to a sequence --

local SequenceLinear, parent = torch.class('onmt.SequenceLinear','onmt.Network')

function SequenceLinear:__init(inputDim, outputDim, bias)
	
	-- assume that the input is a 3D sequence
	self.inputViewer = nn.View(1,1, -1):setNumInputDims(3)
	-- we reshape it to have the size (batch_size * seq_length) x hidden

	self.outputViewer = nn.View(1, -1):setNumInputDims(2)
	
	parent.__init(self, self:_buildModel(inputDim, outputDim, bias))
end

--build the nn Graph. 
function SequenceLinear:_buildModel(inputDim, outputDim, bias)
	
	-- 3D tensor: batch_size * seq_length * inputDim
	local input = nn.Identity()()
	
	local inputs = {input}
	
	local viewedInput = self.inputViewer(input)

	local transformedInput = nn.Linear(inputDim, outputDim, bias)(viewedInput)
	-- this one has size (batch_size * seq_length) x outputDim
	
	local output = self.outputViewer(transformedInput)
	
	return nn.gModule(inputs, {output})
	
end

function SequenceLinear:updateOutput(input)

	local batchSize = input:size(1)
	local seqLength = input:size(2)
	local inputDim = input:size(3)
	
	self.inputViewer:resetSize(batchSize * seqLength, -1)
	self.outputViewer:resetSize(batchSize, seqLength, -1)
	
	self.output = self.net:updateOutput(input)
	
	return self.output
	
	
end


-- updateGradInput should be the same
