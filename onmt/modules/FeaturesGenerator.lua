--[[ Feature decoder generator. Given RNN state, produce categorical distribution over
tokens and features.

  Implements $$[softmax(W^1 h + b^1), softmax(W^2 h + b^2), ..., softmax(W^n h + b^n)] $$.
--]]
local FeaturesGenerator, parent = torch.class('onmt.FeaturesGenerator', 'onmt.Network')

--[[
Parameters:

  * `rnnSize` - Input rnn size.
  * `outputSizes` - Table of each output size.
--]]
function FeaturesGenerator:__init(rnnSize, outputSizes)
  parent.__init(self, self:_buildGenerator(rnnSize, outputSizes))
end

function FeaturesGenerator:_buildGenerator(rnnSize, outputSizes)
  local generator = nn.ConcatTable()

  for i = 1, #outputSizes do
    generator:add(nn.Sequential()
                    :add(nn.Linear(rnnSize, outputSizes[i]))
                    :add(nn.LogSoftMax()))
  end

  return generator
end
