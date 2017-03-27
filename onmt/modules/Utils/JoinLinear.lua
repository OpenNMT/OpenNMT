-- Join two vectors and transform into one --

local JoinLinear, parent = torch.class('onmt.JoinLinear','onmt.Network')

function JoinLinear:__init(dim, activation, useBias)
	
	self.dim = dim
	self.activation = activation or nn.Tanh
	self.useBias = useBias or false
	parent.__init(self, self:_buildModel(self.dim, self.activation))
end

--build the nn Graph. 
-- Simply concatenate two vectors, 
-- Then use a linear transformation, with an activation followed
function JoinLinear:_buildModel(dim, activation)
	
	local inputs = {}
	
	table.insert(inputs, nn.Identity()())
	table.insert(inputs, nn.Identity()())
	
	local joined = nn.JoinTable(2)(inputs)
	local transformed = nn.Linear(dim*2, dim, self.useBias)(joined)
	
	local output = activation()(transformed)
	
	return nn.gModule(inputs, {output})
	

end
