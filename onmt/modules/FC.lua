require('nngraph')

--[[
 Stack of fully connected modules
--]]
local FC, parent = torch.class('onmt.FC', 'onmt.Network')

--[[
Parameters:

  * `layers` - Number of FC layers, L.
  * `inputSize` - Size of input layer
  * `dropout` - Dropout rate to use (in $$[0,1]$$ range).
  * `residual` - Residual connections between layers.
--]]
function FC:__init(layers, inputSize, dropout, residual)
  dropout = dropout or 0

  self.dropout = dropout

  parent.__init(self, self:_buildModel(layers, inputSize, dropout, residual))
end

--[[ Stack the FC units. ]]
function FC:_buildModel(layers, inputSize, dropout, residual)
  local inputs = {}
  local outputs = {}

  table.insert(inputs, nn.Identity()()) -- x: batchSize x inputSize
  local h = inputs[#inputs]
  local hs = {}

  for L = 1, layers do
    h = nn.Dropout(dropout)(h)
    h = nn.Linear(inputSize, inputSize)(h)
    if residual and L > 2 then
      h = nn.CAddTable()({h, hs[#hs-1]})
    end
    if L < layers then
      h = nn.ReLU()(h)
    else
      h = nn.Tanh()(h)
    end
    table.insert(hs, h)
  end

  table.insert(outputs, h)

  return nn.gModule(inputs, outputs)
end

