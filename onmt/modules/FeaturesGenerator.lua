--[[ Feature decoder generator. Given RNN state, produce categorical distribution over
tokens and features.

  Implements $$[softmax(W^1 h + b^1), softmax(W^2 h + b^2), ..., softmax(W^n h + b^n)] $$.
--]]
local FeaturesGenerator, parent = torch.class('onmt.FeaturesGenerator', 'onmt.Network')

--[[
Parameters:

  * `rnnSize` - Input rnn size.
  * `outputSize` - Output size (number of tokens).
  * `features` - table of feature sizes.
--]]
function FeaturesGenerator:__init(rnnSize, outputSize, features)
  parent.__init(self, self:_buildGenerator(rnnSize, outputSize, features))
end

function FeaturesGenerator:_buildGenerator(rnnSize, outputSize, features)
  local generator = nn.ConcatTable()

  -- Add default generator.
  generator:add(nn.Sequential()
                  :add(onmt.Generator(rnnSize, outputSize))
                  :add(nn.SelectTable(1)))

  -- Add a generator for each target feature.
  for i = 1, #features do
    generator:add(nn.Sequential()
                    :add(nn.Linear(rnnSize, features[i]:size()))
                    :add(nn.LogSoftMax()))
  end

  return generator
end
