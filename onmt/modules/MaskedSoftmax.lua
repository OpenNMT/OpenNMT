require('nngraph')

--[[ A batched-softmax wrapper to mask the probabilities of padding.

  For instance there may be a batch of instances where A is padding.

    AAXXXX
    AAAAXX
    XXXXXX

  MaskedSoftmax ensures that no probability is given to the A's.

  For this example, `sourceSizes` is {4, 2, 6} and `sourceLength` is 6.
--]]
local MaskedSoftmax, parent = torch.class('onmt.MaskedSoftmax', 'onmt.Network')


--[[ A nn-style module that applies a softmax on input that gives no weight to the left padding.

Parameters:

  * `sourceSizes` -  the true lengths (with left padding).
  * `sourceLength` - the length of the batch.
--]]
function MaskedSoftmax:__init(sourceSizes, sourceLength)
  parent.__init(self, self:_buildModel(sourceSizes, sourceLength))
end

function MaskedSoftmax:_buildModel(sourceSizes, sourceLength)
  local numSents = sourceSizes:size(1)
  local input = nn.Identity()()
  local softmax = nn.SoftMax()(input) -- numSents x State.sourceLength

  -- Now we are masking the part of the output we don't need
  local tab = nn.SplitTable(1)(softmax) -- numSents x { State.sourceLength }
  local par = nn.ParallelTable()

  for b = 1, numSents do
    local padLength = sourceLength - sourceSizes[b]

    local seq = nn.Sequential()
    seq:add(nn.Narrow(1, padLength + 1, sourceSizes[b]))
    seq:add(nn.Padding(1, -padLength, 1, 0))
    par:add(seq)
  end

  local outTab = par(tab) -- numSents x { State.sourceLength }
  local output = nn.JoinTable(1)(outTab) -- numSents x State.sourceLength
  output = nn.View(numSents, sourceLength)(output)

  -- Make sure the vector sums to 1 (softmax output)
  output = nn.Normalize(1)(output)

  return nn.gModule({input}, {output})
end
