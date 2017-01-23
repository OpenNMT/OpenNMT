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

--[[Helper function. Recursively convert flat `batchSize * beamSize` tensors
 to 2D `(batchSize, beamSize)` tensors.

Parameters:

  * `v` - flat tensor of size `batchSize * beamSize` or a table containing such tensors.
  * `beamSize` - beam size

Returns: `(batchSize, beamSize)` tensor or a table containing such tensors.

--]]
local function flatToRc(v, beamSize)
  return onmt.utils.Tensor.recursiveApply(v, function (h)
    local batchSize = math.floor(h:size(1) / beamSize)
    local sizes = {}
    for j = 2, #h:size() do
        sizes[j - 1] = h:size(j)
    end
    return h:view(batchSize, beamSize, table.unpack(sizes))
  end)
end

--[[Helper function. Recursively convert 2D `(batchSize, beamSize)` tensors to
 flat `batchSize * beamSize` tensors.

Parameters:

  * `v` - flat tensor of size `(batchSize, beamSize)` or a table containing such tensors.
  * `beamSize` - beam size

Returns: `batchSize * beamSize` tensor or a table containing such tensors.

--]]
local function rcToFlat(v)
  return onmt.utils.Tensor.recursiveApply(v, function (h)
    local sizes = {}
    sizes[1] = h:size(1) * h:size(2)
    for j = 3, #h:size() do
        sizes[j - 1] = h:size(j)
    end
    return h:view(table.unpack(sizes))
  end)
end

--[[Helper function. Recursively select `(batchSize, ...)` tensors by
  specified batch indexes.

Parameters:

  * `v` - tensor of size `(batchSize, ...)` or a table containing such tensors
  * `indexes` - a table of the desired batch indexes

Returns: Indexed `(newBatchSize, ...)` tensor or a table containing such tensors

--]]
local function selectBatch(v, remaining)
  return onmt.utils.Tensor.recursiveApply(v, function (h)
    if not torch.isTensor(remaining) then
      remaining = torch.LongTensor(remaining)
    end
    return h:index(1, remaining)
  end)
end

--[[Helper function. Recursively select `(batchSize * beamSize, ...)` tensors by
  specified batch index and beam index.

Parameters:

  * `v` - tensor of size `(batchSize * beamSize, ...)` or a table containing such tensors
  * `beamSize` - beam size
  * `batch` - the desired batch index
  * `beam` - the desired beam index

Returns: Indexed `(...)` tensor or a table containing such tensors

--]]
local function selectBatchBeam(v, beamSize, batch, beam)
  return onmt.utils.Tensor.recursiveApply(v, function (h)
    local batchSize = math.floor(h:size(1) / beamSize)
    local sizes = {}
    for j = 2, #h:size() do
        sizes[j - 1] = h:size(j)
    end
    local hOut = h:view(batchSize, beamSize, table.unpack(sizes))
    return hOut[{batch, beam}]
  end)
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

function Beam:nextBeam(token, scores, backPointers, beamSize)
  local newBeam = Beam.new(self:nextTokens(token, backPointers, beamSize),
                           self:nextState(backPointers, beamSize))
  newBeam:setScores(scores)
  return newBeam
end

function Beam:remaining2Orig(b)
  if b then
    return self._remaining2Orig[b]
  else
    return self._remaining2Orig
  end
end

function Beam:orig2Remaining(b)
  if b then
    return self._orig2Remaining[b]
  else
    return self._orig2Remaining
  end
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

function Beam:removeFinishedBatches(remainingIds, beamSize)
  if #remainingIds > 0 then
    self._state = rcToFlat(selectBatch(
      flatToRc(self._state, beamSize), remainingIds))
    self._tokens = rcToFlat(selectBatch(
      flatToRc(self._tokens, beamSize), remainingIds))
    self._scores = selectBatch(self._scores, remainingIds)
  end
end
