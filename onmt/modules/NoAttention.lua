require('nngraph')

--[[ No attention module

--]]
local NoAttention, parent = torch.class('onmt.NoAttention', 'onmt.Network')

--[[A nn-style module computing attention.

  Parameters:

  * `dim` - dimension of the context vectors.
--]]
function NoAttention:__init(_, dim)
  parent.__init(self, self:_buildModel(dim))
end

function NoAttention:_buildModel(dim)
  local inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())

  local lastContext = nn.Select(2,-1)(inputs[2])
  local contextCombined = nn.JoinTable(2)({lastContext, inputs[1]}) -- batchL x dim*2
  local contextOutput = nn.Tanh()(nn.Linear(dim*2, dim, false)(contextCombined))

  return nn.gModule(inputs, {contextOutput})
end
