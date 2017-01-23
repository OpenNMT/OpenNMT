local Beam = torch.class('Beam')

--[[Helper function.

Parameters:

  * `v` - tensor
  * `x` - reference tensor

Returns: if `x` is cuda tensor, return `v:cuda()`; otherwise, return `v`.

--]]
local function localize(v, x)
  if string.match(x:type(), 'Cuda') then
    return v:cuda()
  else
    return v
  end
end

--[[Helper function. Recursively index `(batchSize * beamSize, ...)`
  tensors by specified indexes.

Parameters:

  * `v` - tensor of size `(batchSize * beamSize, ...)` or a table containing such tensors
  * `indexes` - a tensor of size `(batchSize, k)` specifying the desired indexes
  * `beamSize` - beam size

Returns: Indexed `(batchSize * k, ...)` tensor or a table containing such tensors

--]]
local function selectBeam(v, indexes, beamSize)
  return onmt.utils.Tensor.recursiveApply(v, function (h)
    local batchSize = indexes:size(1)
    local k = indexes:size(2)
    beamSize = beamSize or k
    return h:index(1, indexes:view(-1):long()
                 + (torch.range(0, (batchSize - 1) * beamSize, beamSize):long())
                 :contiguous():view(batchSize, 1)
                 :expandAs(indexes):contiguous():view(-1))
  end)
end

function Beam:__init(token, state, batchSize)
  batchSize = batchSize or token:size(1)
  self._remaining = batchSize

  self._tokens = { token }
  self._state = state

  self._scores = localize(torch.zeros(batchSize), token)
  self._backPointers = nil
  self._orig2Remaining = {}
  self._remaining2Orig = {}
  for b = 1, batchSize do
    self._orig2Remaining[b] = b
    self._remaining2Orig[b] = b
  end
end

--[[Helper function. Recursively expand `(batchSize, ...)` tensors
  to `(batchSize * beamSize, ...)` tensors.

Parameters:

  * `v` - tensor of size `(batchSize, ...)` or a table containing such tensors
  * `beamSize` - beam size

Returns: Expanded `(batchSize * beamSize, ...)` tensor or a table containing
  such tensors

--]]
local function replicateBeam(v, beamSize)
  return onmt.utils.Tensor.recursiveApply(v, function (h)
    local batchSize = h:size(1)
    local sizes = {}
    for j = 2, #h:size() do
        sizes[j - 1] = h:size(j)
    end
    return h:contiguous():view(batchSize, 1, table.unpack(sizes))
                  :expand(batchSize, beamSize, table.unpack(sizes)):contiguous()
                  :view(batchSize * beamSize, table.unpack(sizes))
  end)
end

--[[Helper function. In the first step, if there is only 1 hypothesis per
  batch, then each hypothesis is replicated beamSize times to keep consistency
  with the following beam search steps, while the scores of the auxiliary
  hypotheses are set to -inf.

Parameters:

  * `beamSize` - beam size

--]]
function Beam:replicate(beamSize)
  assert (#self._tokens == 1, 'only the first beam may need replicating!')
  local token = self._tokens[1]
  local batchSize = token:size(1)
  self._tokens[1] = replicateBeam(token, beamSize)
  self._state = replicateBeam(self._state, beamSize)
  self._scores = replicateBeam(self._scores, beamSize)
  local maskScores = self._scores.new():resize(batchSize, beamSize)
  maskScores:fill(-math.huge)
  maskScores:select(2,1):fill(0)
  self._scores:add(maskScores:view(-1))
end

function Beam:tokens()
  return self._tokens
end

function Beam:flatTokens()
  return self._tokens:view(-1)
end
function Beam:state()
  return self._state
end
function Beam:setState(state)
  self._state = state
end

function Beam:scores()
  return self._scores
end

function Beam:backPointers()
  return self._backPointers
end

function Beam:expandScores(scores, beamSize)
  local remaining = math.floor(scores:size(1) / beamSize)
  local vocabSize = scores:size(2)
  local expandedScores = (scores:view(remaining, beamSize, -1)
                          + self._scores:view(remaining, beamSize, 1)
                                        :expand(remaining, beamSize, vocabSize)
                         ):view(remaining, -1)
  return expandedScores
end

function Beam:next(token, backPointers, beamSize)
  return Beam.new(self:nextTokens(token, backPointers, beamSize),
                  self:nextState(backPointers, beamSize))
end

-- Select the on-beam states using the pointers
function Beam:nextState(backPointers, beamSize)
  local nextState = selectBeam(self._state, backPointers, beamSize)
  return nextState
end

function Beam:nextTokens(token, backPointers, beamSize)
  local nextTokens = selectBeam(self._tokens, backPointers, beamSize)
  nextTokens[#nextTokens + 1] = token
  return nextTokens
end
-- binary vector
function Beam:batchesFinished()
  return self._batchesFinished
end

-- dict
function Beam:addCompletedHypotheses()
  table.insert(self.completed, {}) -- batch, pointer, t, score, etc.
end
