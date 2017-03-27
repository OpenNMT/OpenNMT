local GRU, parent = torch.class('onmt.GRUModule', 'onmt.Network')

--]]
function GRU:__init(inputSize, hiddenSize, dropout)
  parent.__init(self, self:_buildModel(inputSize, hiddenSize, dropout))
end



function GRU:_buildModel(inputSize, hiddenSize, dropout)
  local inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())

  -- Recurrent input.
  local prevH = inputs[1]
  
  prevH = nn.Dropout(dropout)(prevH)
  
  -- Previous layer input.
  local x = inputs[2]
  
  function LinearSum(iSize, hSize, x, h)
	local i2h = nn.Linear(iSize, hSize)(x)
	local h2h = nn.Linear(hSize, hSize)(h)
	return nn.CAddTable()({i2h, h2h})
  end
  
  local uGate = nn.Sigmoid()(LinearSum(inputSize, hiddenSize, x, prevH))
  
  local rGate = nn.Sigmoid()(LinearSum(inputSize, hiddenSize, x, prevH))
  
  local gatedHidden = nn.CMulTable()({rGate, prevH})
  
  local p2 = nn.Linear(hiddenSize, hiddenSize)(gatedHidden)
  local p1 = nn.Linear(inputSize, hiddenSize)(x)
  
  local hiddenCandidate = nn.Tanh()(nn.CAddTable()({p1,p2}))
  local zh = nn.CMulTable()({uGate, hiddenCandidate})
  local zhm1 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(uGate)), prevH})
  
  local nextH = nn.CAddTable()({zh, zhm1})


  return nn.gModule(inputs, {nextH})
end

return GRU
