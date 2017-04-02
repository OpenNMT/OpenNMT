local LayerNormalization, parent = torch.class('onmt.LayerNormalization','onmt.Network')


function LayerNormalization:__init(dim, biasInit, eps, affine)
	
	self.dim = dim
	self.biasInit = biasInit or nil
	self.affine = affine or true
	self.eps = eps or 1e-5
	parent.__init(self, self:_buildModel(self.dim, self.eps, self.affine))
end

--build the nn Graph. 
function LayerNormalization:_buildModel(dim, eps, affine)
	
	local input = nn.Identity()()
	
	local mean = nn.Mean(2)(input)
	
	local mean_rep = nn.Replicate(dim, 2)(mean)
	
	local input_center = nn.CSubTable()({input, mean_rep})
	
	local std = nn.Sqrt()(nn.Mean(2)(nn.Square()(input_center)))
	
	local std_rep = nn.AddConstant(eps)(nn.Replicate(dim,2)(std))
	
	local output = nn.CDivTable()({input_center, std_rep})
	
	if affine == true then
		
		local biasTransform = nn.Add(dim, false)
		self.biasTransform = biasTransform
		
		local gainTransform = nn.CMul(dim)
		self.gainTransform = gainTransform
		
		output = biasTransform(gainTransform(output))
	end
	
	return nn.gModule({input},{output})

end

function LayerNormalization:postParametersInitialization()
  
  if self.affine == true then
	
	if self.biasInit ~= nil then
		self.biasTransform.bias:fill(self.biasInit)
	end
	
	self.gainTransform.weight:fill(1.)
  end
end
