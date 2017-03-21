require('nngraph')

--[[
Implementation of a single stacked-GRU step as
an nn unit.

Computes $$(h_{t-1}, x_t) => (h_{t})$$.

--]]
local ContextCoverage, parent = torch.class('onmt.ContextCoverage', 'onmt.Network')

--[[
Parameters:

  * `hiddenSize' - Size of the hidden layers
  * `coverageSize` - Size of input layer
--]]
function ContextCoverage:__init(hiddenSize, coverageSize)
  
  self.hiddenSize = hiddenSize
  self.coverageSize = coverageSize
  
  self.replicators = {}
  parent.__init(self, self:_buildModel(hiddenSize, coverageSize))
end

--[[ Gated unit for coverage. ]]
function ContextCoverage:_buildModel(hiddenSize, coverageSize)
  -- inputs: { prevOutput L1, ..., prevOutput Ln, input }
  -- outputs: { output L1, ..., output Ln }

  local inputs = {}
  local outputs = {}
  
  table.insert(inputs, nn.Identity()()) -- Last coverage vector
  table.insert(inputs, nn.Identity()()) -- Context matrix
  table.insert(inputs, nn.Identity()()) -- Alignment vector
  table.insert(inputs, nn.Identity()()) -- Decoder hidden layer
  
  local alignment = nn.Replicate(1, 3)(inputs[3]) -- batchSize x sourceLength x 1
  local context = inputs[2]
  local lastCoverage = inputs[1]
  local decHidden = inputs[4]
  
  -- transform the decoder hidden layer
  --~ decHidden = nn.Linear(hiddenSize, coverageSize)(decHidden)
  
  -- In this GRU unit, there are actually three current inputs: context, alignment and hidden layer
  local function buildGate(cov, context, align, hidden)
	local projectedContext = onmt.SequenceLinear(hiddenSize, coverageSize)(context) -- these two will play the role of 'x' in typical GRU
	local projectedAlign = onmt.SequenceLinear(1, coverageSize)(align)
	local projectedCov = onmt.SequenceLinear(coverageSize, coverageSize)(cov)
	
	-- For the hidden layer
	local transformedHidden = nn.Linear(hiddenSize, coverageSize)(hidden) -- transform first
	local replicator = onmt.Replicator(2, 1) -- then replicate
	local projectedHidden = replicator(transformedHidden)
	table.insert(self.replicators, replicator) -- we have to manually set the replicate length later
	
	return nn.CAddTable()({projectedCov, projectedAlign, projectedContext, projectedHidden})
  end
  
  local rGate = nn.Sigmoid()(buildGate(lastCoverage, context, alignment, decHidden)) -- reset gate
  
  local uGate = nn.Sigmoid()(buildGate(lastCoverage, context, alignment, decHidden)) -- update gate
  
  local gatedH = nn.CMulTable()({rGate, lastCoverage})
  
  local candidateH = nn.Tanh()(buildGate(gatedH, context, alignment, decHidden))
  
  
  -- compute new interpolated hidden state, based on the update gate
  local zh = nn.CMulTable()({uGate, candidateH})
  
  local zhm1 = nn.CMulTable()({nn.AddConstant(1, false)(nn.MulConstant(-1, false)(uGate)), lastCoverage})
  
  local nextCoverage = nn.CAddTable()({zh, zhm1})
  
  
  return nn.gModule(inputs, {nextCoverage})
 
end

-- When we update the input, we have to manually change the length of the replicator to match the sequence length
function ContextCoverage:updateOutput(input)

	local lastCov, context, align, hidden = unpack(input)
	local sourceLength = context:size(2)
		
	for _, replicator in pairs(self.replicators) do
		replicator:setNFeatures(sourceLength)
	end
	
	self.output = self.net:updateOutput({lastCov, context, align, hidden})
	
	return self.output
	
end


