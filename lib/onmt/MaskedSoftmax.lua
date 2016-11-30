require 'nngraph'

--[[ A batched-softmax wrapper to mask the probabilities of padding.

    AXXXAA  
    AXXAAA
    AXXXXX
 
--]]
local MaskedSoftmax, parent = torch.class('onmt.MaskedSoftmax', 'nn.Container')


--[[ A nn-style module that applies a softmax on input that gives no weight to the left padding.

Parameters:

  * `source_sizes` -  the true lengths (with left padding).
  * `source_length` - the max length in the batch `beam_size`.
  * `beam_size` - beam size ${K}
--]]
function MaskedSoftmax:__init(source_sizes, source_length, beam_size)
  parent.__init(self)
  --TODO: better names for these variables. Beam size =? batch_size?
  self.net = self:_buildModel(source_sizes, source_length, beam_size)
  self:add(self.net)
end

function MaskedSoftmax:_buildModel(source_sizes, source_length, beam_size)

  local num_sents = source_sizes:size(1)
  local input = nn.Identity()()
  local softmax = nn.SoftMax()(input) -- beam_size*num_sents x State.source_length

  -- Now we are masking the part of the output we don't need
  local tab
  if beam_size ~= nil then
    tab = nn.SplitTable(2)(nn.View(beam_size, num_sents, source_length)(softmax))
    -- num_sents x { beam_size x State.source_length }
  else
    tab = nn.SplitTable(1)(softmax) -- num_sents x { State.source_length }
  end

  local par = nn.ParallelTable()

  for b = 1, num_sents do
    local pad_length = source_length - source_sizes[b]
    local dim = 2
    if beam_size == nil then
      dim = 1
    end

    local seq = nn.Sequential()
    seq:add(nn.Narrow(dim, pad_length + 1, source_sizes[b]))
    seq:add(nn.Padding(1, -pad_length, 1, 0))
    par:add(seq)
  end

  local out_tab = par(tab) -- num_sents x { beam_size x State.source_length }
  local output = nn.JoinTable(1)(out_tab) -- num_sents*beam_size x State.source_length
  if beam_size ~= nil then
    output = nn.View(num_sents, beam_size, source_length)(output)
    output = nn.Transpose({1,2})(output) -- beam_size x num_sents x State.source_length
    output = nn.View(beam_size*num_sents, source_length)(output)
  else
    output = nn.View(num_sents, source_length)(output)
  end

  -- Make sure the vector sums to 1 (softmax output)
  output = nn.Normalize(1)(output)

  return nn.gModule({input}, {output})
end

function MaskedSoftmax:updateOutput(input)
  self.output = self.net:updateOutput(input)
  return self.output
end

function MaskedSoftmax:updateGradInput(input, gradOutput)
  return self.net:updateGradInput(input, gradOutput)
end

function MaskedSoftmax:accGradParameters(input, gradOutput, scale)
  return self.net:accGradParameters(input, gradOutput, scale)
end
