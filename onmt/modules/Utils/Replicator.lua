local Replicator, parent = torch.class('onmt.Replicator','nn.Replicate')

function Replicator:__init(nf, dim, ndim)

	parent.__init(self, nf, dim, ndim)
	
end

-- we have to manually set this length according to each minibatch 
function Replicator:setNFeatures(nf)

	self.nfeatures = nf
end
