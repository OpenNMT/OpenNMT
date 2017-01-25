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
    local sizes = {}
    local ones = {}
    for j = 2, #h:size() do
        sizes[j - 1] = h:size(j)
        ones[j - 1] = 1
    end
    return h:view(batchSize, beamSize, table.unpack(sizes))
            :gather(2, indexes:view(batchSize, k, table.unpack(ones))
                              :expand(batchSize, k, table.unpack(sizes)))
            :view(batchSize * k, table.unpack(sizes))
  end)
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


--[[Constructor.

Parameters:

  * `token` - tensor of size `(batchSize, vocabSize)` or `(batchSize * beamSize, vocabSize)`, or a list of such tensors.
  * `state` - an iteratable object, where the contained tensors should have the same first dimension as `token`.
  * `remaining` - remaining batch size. [`token:size(1)`]

--]]
function Beam:__init(token, state, remaining)
  self._remaining = remaining or token:size(1)

  if torch.type(token) == 'table' then
    self._tokens = token
  else
    self._tokens = { token }
  end
  self._state = state

  self._scores = localize(torch.zeros(self._remaining), self._tokens[1])
  self._backPointer = nil
  self._orig2Remaining = {}
  self._remaining2Orig = {}
end

function Beam:tokens()
  return self._tokens
end

function Beam:state()
  return self._state
end

function Beam:remaining()
  return self._remaining
end

function Beam:setState(state)
  self._state = state
end

function Beam:setScores(scores)
  self._scores = scores:view(-1)
end

function Beam:setBackPointer(backPointer)
  self._backPointer = backPointer:view(-1)
end

function Beam:setOrig2Remaining(origId, remainingId)
  self._orig2Remaining[origId] = remainingId
end

function Beam:setRemaining2Orig(remainingId, origId)
  self._remaining2Orig[remainingId] = origId
end

function Beam:scores()
  return self._scores
end

function Beam:backPointer()
  return self._backPointer
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

function Beam:expandScores(scores, beamSize)
  local remaining = math.floor(scores:size(1) / beamSize)
  local vocabSize = scores:size(2)
  local expandedScores = (scores:view(remaining, beamSize, -1)
                          + self._scores:view(remaining, beamSize, 1)
                                        :expand(remaining, beamSize, vocabSize)
                         ):view(remaining, -1)
  return expandedScores
end

function Beam:nextBeam(token, scores, backPointer, beamSize)
  local remaining = math.floor(token:size(1) / beamSize)
  local newBeam = Beam.new(self:nextTokens(token, backPointer, beamSize),
                           self:nextState(backPointer, beamSize), remaining)
  newBeam:setScores(scores)
  newBeam:setBackPointer(backPointer)
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
function Beam:nextState(backPointer, beamSize)
  local nextState = selectBeam(self._state, backPointer, beamSize)
  return nextState
end

function Beam:nextTokens(token, backPointer, beamSize)
  local nextTokens = selectBeam(self._tokens, backPointer, beamSize)
  nextTokens[#nextTokens + 1] = token
  return nextTokens
end

-- binary vector
function Beam:batchesFinished()
  return self._batchesFinished
end

function Beam:removeFinishedBatches(remainingIds, beamSize)
  self._remaining = #remainingIds
  if #remainingIds > 0 then
    self._state = rcToFlat(selectBatch(
                           flatToRc(self._state, beamSize), remainingIds))
    self._tokens = rcToFlat(selectBatch(
                           flatToRc(self._tokens, beamSize), remainingIds))
    self._scores = rcToFlat(selectBatch(
                           flatToRc(self._scores, beamSize), remainingIds))
    self._backPointer = rcToFlat(selectBatch(
                           flatToRc(self._backPointer, beamSize), remainingIds))
  end
end

function Beam:indexState(beamSize, batchId, beamId, keptIndexes)
  keptIndexes = keptIndexes or {}
  local keptState = {}
  for _, val in pairs(keptIndexes) do
    keptState[val] = self._state[val]
  end
  return selectBatchBeam(keptState, beamSize, batchId, beamId)
end

function Beam:indexToken(beamSize, batchId, beamId)
  local token = self._tokens[#self._tokens]
  return selectBatchBeam(token, beamSize, batchId, beamId)
end

function Beam:indexBackPointer(beamSize, batchId, beamId)
  if self._backPointer then
    return selectBatchBeam(self._backPointer, beamSize, batchId, beamId)
  end
end

function Beam.addCompletedHypotheses(score, tok, bp, t, batchId)
  local hypothesis = {score, tok, bp, t}
  Beam._completed = Beam._completed or {}
  Beam._completed[batchId] = Beam._completed[batchId] or {}
  -- Maintain a sorted list.
  local id = #Beam._completed[batchId] + 1
  Beam._completed[batchId][id] = hypothesis
  while id > 1 do
    if Beam._completed[batchId][id - 1][1] < score then
      Beam._completed[batchId][id - 1], Beam._completed[batchId][id] =
                 Beam._completed[batchId][id], Beam._completed[batchId][id - 1]
      id = id - 1
    else
      break
    end
  end
end

function Beam.completed(batchId)
  if Beam._completed then
    return Beam._completed[batchId] or {}
  else
    return {}
  end
end

function Beam.removeCompleted(batchId)
  if Beam._completed then
    Beam._completed[batchId] = nil
  end
end

return Beam
