--[[ Class for maintaining statistics of each step. A beam mainly consists of
  a list of tokens and a state. Tokens are stored as flat int tensors of size
  `numRemaining * beamSize`, while state can be either a tensor with first dimension size
  `batchSize`, or an iterable object containing several such tensors.
--]]
local Beam = torch.class('Beam')

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
    return h
      :contiguous()
      :view(batchSize, beamSize, table.unpack(sizes))
      :gather(2, indexes
                :view(batchSize, k, table.unpack(ones))
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
    return h
      :contiguous()
      :view(batchSize, 1, table.unpack(sizes))
      :expand(batchSize, beamSize, table.unpack(sizes))
      :contiguous()
      :view(batchSize * beamSize, table.unpack(sizes))
  end)
end


--[[Constructor. We allow users to either specify all initial hypotheses by
  passing in `token` and `state` with first dimension `batchSize * beamSize`
  such that there are `beamSize` initial hypotheses for every sequence in the
  batch and pass in the number of sequences `remaining`, or only specify one
  hypothesis per sequence by passing `token` and `state` with first dimension
  `batchSize`, and then `onmt.translate.BeamSearcher` will pad with auxiliary
  hypotheses with scores `-inf` such that each sequence starts with `beamSize`
  hypotheses as in the former case.

Parameters:

  * `token` - tensor of size `(batchSize, vocabSize)` (if start with one initial hypothesis per sequence) or `(batchSize * beamSize, vocabSize)` (if start with `beamSize` initial hypotheses per sequence), or a list of such tensors.
  * `state` - an iterable object, where the contained tensors should have the same first dimension as `token`.
  * `remaining` - (optional) remaining number of sentences. Only necessary if start with `beamSize` hypotheses per sequence. [`token:size(1)`]

--]]
function Beam:__init(token, state, remaining)
  self._remaining = remaining or token:size(1)

  if torch.type(token) == 'table' then
    self._tokens = token
  else
    self._tokens = { token }
  end
  self._state = state

  self._scores = torch.zeros(self._remaining)
  self._backPointer = nil
  self._prevBeam = nil
  self._orig2Remaining = {}
  self._remaining2Orig = {}
  self._step = 1
end

--[[

Returns:

  * `tokens` - a list of tokens. Note that the start-of-sequence symbols is included.

--]]
function Beam:getTokens()
  return self._tokens
end

--[[

Returns:

  * `state` - an abstract iterable object as passed by constructor.

--]]
function Beam:getState()
  return self._state
end

--[[

Returns:

  * `scores` - a flat tensor storing the total scores for each batch.

--]]
function Beam:getScores()
  return self._scores
end

--[[

Returns:

  * `backPointer` - a flat tensor storing the total scores for each batch.

--]]
function Beam:getBackPointer()
  return self._backPointer
end

--[[ Get the number of unfinished sequences. The finished sequences will be
  removed from batch.

Returns:

  * `remaining` - the number of unfinished sequences.

--]]
function Beam:getRemaining()
  return self._remaining
end

--[[ Set state.
--]]
function Beam:setState(state)
  self._state = state
end

--[[ Set scores.
--]]
function Beam:setScores(scores)
  self._scores = scores:view(-1)
end

--[[ Set backPointer.
--]]
function Beam:setBackPointer(backPointer)
  self._backPointer = backPointer:view(-1)
end

-- In the first step, if there is only 1 hypothesis per
-- batch, then each hypothesis is replicated beamSize times to keep consistency
-- with the following beam search steps, while the scores of the auxiliary
-- hypotheses are set to -inf.
function Beam:_replicate(beamSize)
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

function Beam:_expandScores(scores, beamSize)
  local remaining = math.floor(scores:size(1) / beamSize)
  local vocabSize = scores:size(2)
  self._scores = self._scores:typeAs(scores)
  local expandedScores
    = (scores:typeAs(self._scores):view(remaining, beamSize, -1)
         + self._scores:view(remaining, beamSize, 1):expand(remaining, beamSize, vocabSize)
      ):view(remaining, -1)
  return expandedScores
end

function Beam:nextBeam(token, scores, backPointer, beamSize)
  local remaining = math.floor(token:size(1) / beamSize)
  local newBeam = Beam.new(self:nextTokens(token, backPointer, beamSize),
                           self:nextState(backPointer, beamSize),
                           remaining)
  newBeam:setScores(scores)
  newBeam:setBackPointer(backPointer)
  newBeam._prevBeam = self
  newBeam._step = self._step + 1
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
    self._state = rcToFlat(selectBatch(flatToRc(self._state, beamSize), remainingIds))
    self._tokens = rcToFlat(selectBatch(flatToRc(self._tokens, beamSize), remainingIds))
    self._scores = rcToFlat(selectBatch(flatToRc(self._scores, beamSize), remainingIds))
    self._backPointer = rcToFlat(selectBatch(flatToRc(self._backPointer, beamSize), remainingIds))
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

function Beam:cleanUp(keptIndexes)
  keptIndexes = keptIndexes or {}
  local keptState = {}
  for _, val in pairs(keptIndexes) do
    keptState[val] = self._state[val]
  end
  self._state = keptState
end

function Beam:_addCompletedHypotheses(batchId, completed)
  local origId = self:_getOrigId(batchId)
  self._remainingId = self._remainingId or 0
  self._remainingId = self._remainingId + 1
  self._orig2Remaining[origId] = self._remainingId
  self._remaining2Orig[self._remainingId] = origId
  completed = completed:view(self._remaining, -1)
  local scores = self._scores:view(self._remaining, -1)
  local tokens = self._tokens[#self._tokens]:view(self._remaining, -1)
  local backPointers = self._backPointer:view(self._remaining, -1)

  Beam._completed = Beam._completed or {}
  Beam._completed[origId] = Beam._completed[origId] or {}
  for k = 1, completed:size(2) do
    if completed[batchId][k] == 1 then
      local token = tokens[batchId][k]
      local backPointer = backPointers[batchId][k]
      local score = scores[batchId][k]
      local hypothesis = {origId, score, token, backPointer, self._step}

      -- Maintain a sorted list.
      local id = #Beam._completed[origId] + 1
      Beam._completed[origId][id] = hypothesis
      while id > 1 do
        if Beam._completed[origId][id - 1][1] < score then
          Beam._completed[origId][id - 1], Beam._completed[origId][id]
            = Beam._completed[origId][id], Beam._completed[origId][id - 1]
          id = id - 1
        else
          break
        end
      end
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

function Beam:_getOrigId(remainingId)
  local origId
  if self._step <= 2 then
    origId = remainingId
  else
    origId = self._prevBeam:remaining2Orig(remainingId)
  end
  return origId
end

-- Get nBest hypotheses for a particular sequence in the batch.
function Beam:_getTopHypotheses(remainingId, nBest, completed)
  local origId = self:_getOrigId(remainingId)

  -- Get previously completed hypotheses of the sequence.
  local prevCompleted = Beam.completed(origId)

  local hypotheses = {}
  local prevId = 1
  local currId = 1
  completed = completed:view(self._remaining, -1)
  local scores = self._scores:view(self._remaining, -1)
  local tokens = self._tokens[#self._tokens]:view(self._remaining, -1)
  local backPointers = self._backPointer:view(self._remaining, -1)
  for _ = 1, nBest do
    local hypothesis, onBeam, id, finished
    if prevId <= #prevCompleted and prevCompleted[prevId][2] > scores[remainingId][currId] then
      hypothesis = prevCompleted[prevId]
      finished = true
      id = prevId
      onBeam = false
      prevId = prevId + 1
    else
      finished = (completed[remainingId][currId] == 1)
      if finished then
        local score = scores[remainingId][currId]
        local token = tokens[remainingId][currId]
        local backPointer = backPointers[remainingId][currId]
        hypothesis = {origId, score, token, backPointer, self._step}
      end
      id = currId
      onBeam = true
      currId = currId + 1
    end
    table.insert(hypotheses, {onBeam = onBeam, hypothesis = hypothesis, id = id, finished = finished})
  end
  return hypotheses
end
return Beam
