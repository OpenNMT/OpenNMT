--[[ Class for maintaining statistics of each step. A beam mainly consists of
  a list of tokens `tokens` and a state `state`. `tokens[t]` stores a flat tensor
  of size `batchSize * beamSize` representing the tokens at step `t`, while
  `state` can be either a tensor with first dimension size `batchSize * beamSize`,
  or an iterable object containing several such tensors.
--]]
local Beam = torch.class('Beam')

--[[Helper function. Recursively convert flat `batchSize * beamSize` tensors
 to 2D `(batchSize, beamSize)` tensors.

Parameters:

  * `v` - flat tensor of size `batchSize * beamSize` or a table containing such
  tensors.
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

  * `v` - flat tensor of size `(batchSize, beamSize)` or a table containing such
  tensors.
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

  * `v` - tensor of size `(batchSize * beamSize, ...)` or a table containing
  such tensors.
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

  * `v` - tensor of size `(batchSize * beamSize, ...)` or a table containing
  such tensors.
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
  batch and pass in the number of sequences `batchSize`, or only specify one
  hypothesis per sequence by passing `token` and `state` with first dimension
  `batchSize`, and then `onmt.translate.BeamSearcher` will pad with auxiliary
  hypotheses with scores `-inf` such that each sequence starts with `beamSize`
  hypotheses as in the former case.

Parameters:

  * `token` - tensor of size `(batchSize, vocabSize)` (if start with one initial
  hypothesis per sequence) or `(batchSize * beamSize, vocabSize)` (if start with
  `beamSize` initial hypotheses per sequence), or a list of such tensors.
  * `state` - an iterable object, where the contained tensors should have the
  same first dimension as `token`.
  * `batchSize` - optional, number of sentences. Only necessary if
  start with `beamSize` hypotheses per sequence. [`token:size(1)`]

--]]
function Beam:__init(token, state, params, batchSize)
  self._remaining = batchSize or token:size(1)

  if torch.type(token) == 'table' then
    self._tokens = token
  else
    self._tokens = { token }
  end
  self._state = state

  self._params = {}
  if params then
    self._params = params
  else
    self._params.length_norm = 0.0
    self._params.coverage_norm = 0.0
    self._params.eos_norm = 0.0
  end

  self._scores = torch.zeros(self._remaining)
  self._backPointer = nil
  self._prevBeam = nil
  self._orig2Remaining = {}
  self._remaining2Orig = {}
  self._step = 1
end

--[[

Returns:

  * `tokens` - a list of tokens. Note that the start-of-sequence symbols are
  also included. `tokens[t]` stores the tokens at step `t`, which is a tensor
  of size `batchSize * beamSize`.

--]]
function Beam:getTokens()
  return self._tokens
end

--[[

Returns:

  * `state` - an abstract iterable object as passed by constructor. Every tensor
  inside the `state` has first dimension `batchSize * beamSize`

--]]
function Beam:getState()
  return self._state
end

--[[

Returns:

  * `scores` - a flat tensor storing the total scores for each batch. It is of
  size `batchSize * beamSize`.

--]]
function Beam:getScores()
  return self._scores
end

--[[

Returns:

  * `backPointer` - a flat tensor storing the backpointers for each batch. It is
  of size `batchSize * beamSize`

--]]
function Beam:getBackPointer()
  return self._backPointer
end

--[[ Returns the number of unfinished sequences. The finished sequences will be
  removed from batch.

Returns:

  * `remaining` - the number of unfinished sequences.

--]]
function Beam:getRemaining()
  return self._remaining
end

--[[ Since finished sequences are being removed from the batch, this function
  provides a way to convert the remaining batch id to the original batch id.

--]]
function Beam:remaining2Orig(remainingId)
  if remainingId then
    return self._remaining2Orig[remainingId]
  else
    return self._remaining2Orig
  end
end

--[[ Since finished sequences are being removed from the batch, this function
  provides a way to convert the original batch id to the remaining batch id.

--]]
function Beam:orig2Remaining(origId)
  if origId then
    return self._orig2Remaining[origId]
  else
    return self._orig2Remaining
  end
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

-- In the first step, if there is only 1 hypothesis per batch, then each
-- hypothesis is replicated beamSize times to keep consistency with the
-- following beam search steps, while the scores of the auxiliary hypotheses
-- are set to -inf.
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

-- Normalize scores by length and coverage
function Beam:_normalizeScores(scores)

  if #self._state ~= 8 then
    return scores
  end

  local function normalizeLength(t)
    local alpha = self._params.length_norm
    local norm_term = math.pow((5.0 + t)/6.0, alpha)
    return norm_term
  end

  local function normalizeCoverage(ap)
    local beta = self._params.coverage_norm
    local result = torch.cmin(ap, 1.0):log():sum(3):mul(beta)
    return result
  end

  local normScores = scores

  if self._params.length_norm ~= 0 then
    local step = self._step
    local lengthPenalty = normalizeLength(step)
    normScores = torch.div(normScores, lengthPenalty)
  end

  if self._params.coverage_norm ~= 0 then
    local cumAttnProba = self._state[8]:view(self._remaining, scores:size(2), -1)
    local coveragePenalty = normalizeCoverage(cumAttnProba)

    if (scores:nDimension() > 2) then
      coveragePenalty =  coveragePenalty:expand(scores:size())
    else
      coveragePenalty = coveragePenalty:viewAs(scores)
    end
    normScores = torch.add(normScores, coveragePenalty)
  end

  return normScores

end


-- Given new scores, combine that with the previous total scores and find the
-- top K hypotheses to form the next beam.
function Beam:_expandScores(scores, beamSize)
  local remaining = math.floor(scores:size(1) / beamSize)
  local vocabSize = scores:size(2)

  if #self._state == 8 and self._params.eos_norm > 0 then
    local EOS_penalty = torch.div(self._state[6]:view(remaining, beamSize), self._step/self._params.eos_norm)
    scores:view(remaining, beamSize, -1)[{{},{},onmt.Constants.EOS}]:cmul(EOS_penalty)
  end

  self._scores = self._scores:typeAs(scores)
  local expandedScores
    = (scores:typeAs(self._scores):view(remaining, beamSize, -1)
         + self._scores:view(remaining, beamSize, 1):expand(remaining, beamSize, vocabSize)
      )

  local normExpandedScores = self:_normalizeScores(expandedScores)
  return expandedScores:view(remaining, -1), normExpandedScores:view(remaining, -1)
end

-- Create a new beam given new token, scores and backpointer.
function Beam:_nextBeam(token, scores, backPointer, beamSize)
  local remaining = math.floor(token:size(1) / beamSize)
  local params = self._params
  local newBeam = Beam.new(self:_nextTokens(token, backPointer, beamSize),
                           self:_nextState(backPointer, beamSize),
                           params,
                           remaining)
  newBeam:setScores(scores)
  newBeam:setBackPointer(backPointer)
  newBeam._prevBeam = self
  newBeam._step = self._step + 1
  return newBeam
end

-- Select the on-beam states using the pointers
function Beam:_nextState(backPointer, beamSize)
  local nextState = selectBeam(self._state, backPointer, beamSize)
  return nextState
end

-- Given backpointers, index the tokens in the history to form the next beam's
-- token list.
function Beam:_nextTokens(token, backPointer, beamSize)
  local nextTokens = selectBeam(self._tokens, backPointer, beamSize)
  nextTokens[#nextTokens + 1] = token
  return nextTokens
end

-- Remove finished sequences to save computation.
function Beam:_removeFinishedBatches(remainingIds, beamSize)
  self._remaining = #remainingIds
  if #remainingIds > 0 then
    self._state = rcToFlat(selectBatch(flatToRc(self._state, beamSize), remainingIds))
    self._tokens = rcToFlat(selectBatch(flatToRc(self._tokens, beamSize), remainingIds))
    self._scores = rcToFlat(selectBatch(flatToRc(self._scores, beamSize), remainingIds))
    self._backPointer = rcToFlat(selectBatch(flatToRc(self._backPointer, beamSize), remainingIds))
  end
end

-- Index the iterable state object by given batch id and beam id.
function Beam:_indexState(beamSize, batchId, beamId, keptIndexes)
  keptIndexes = keptIndexes or {}
  local keptState = {}
  for _, val in pairs(keptIndexes) do
    keptState[val] = self._state[val]
  end
  return selectBatchBeam(keptState, beamSize, batchId, beamId)
end

-- Index the last step token by given batch id and beam id.
function Beam:_indexToken(beamSize, batchId, beamId)
  local token = self._tokens[#self._tokens]
  return selectBatchBeam(token, beamSize, batchId, beamId)
end

-- Index backpointer by given batch id and beam id.
function Beam:_indexBackPointer(beamSize, batchId, beamId)
  if self._backPointer then
    return selectBatchBeam(self._backPointer, beamSize, batchId, beamId)
  end
end

-- To save memory, only states at keptIndexes will be kept, while the rest
-- are discarded.
function Beam:_cleanUp(keptIndexes)
  keptIndexes = keptIndexes or {}
  local keptState = {}
  for _, val in pairs(keptIndexes) do
    keptState[val] = self._state[val]
  end
  self._state = keptState
end

-- Add completed hypotheses to buffer.
function Beam:_addCompletedHypotheses(batchId, completed)
  local origId = self:_getOrigId(batchId)
  self._remainingId = self._remainingId or 0
  self._remainingId = self._remainingId + 1
  self._orig2Remaining[origId] = self._remainingId
  self._remaining2Orig[self._remainingId] = origId
  completed = completed:view(self._remaining, -1)
  local scores = self._scores:view(self._remaining, -1)
  local normScores = self:_normalizeScores(scores)
  local tokens = self._tokens[#self._tokens]:view(self._remaining, -1)
  local backPointers = self._backPointer:view(self._remaining, -1)

  Beam._completed = Beam._completed or {}
  Beam._completed[origId] = Beam._completed[origId] or {}
  for k = 1, completed:size(2) do
    if completed[batchId][k] == 1 then
      local token = tokens[batchId][k]
      local backPointer = backPointers[batchId][k]
      local normScore = normScores[batchId][k]
      local hypothesis = {origId, normScore, token, backPointer, self._step}

      -- Maintain a sorted list.
      local id = #Beam._completed[origId] + 1
      Beam._completed[origId][id] = hypothesis
      while id > 1 do
        if Beam._completed[origId][id - 1][2] < normScore then
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

-- Free buffer when a sequence is finished.
function Beam._removeCompleted(batchId)
  if Beam._completed then
    Beam._completed[batchId] = nil
  end
end

-- Get the original if of a sequence given its current position in the batch.
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
  local prevCompleted
  if Beam._completed then
    prevCompleted = Beam._completed[origId] or {}
  else
    prevCompleted = {}
  end

  local hypotheses = {}
  local prevId = 1
  local currId = 1
  completed = completed:view(self._remaining, -1)
  local scores = self._scores:view(self._remaining, -1)
  local normScores = self:_normalizeScores(scores)
  local tokens = self._tokens[#self._tokens]:view(self._remaining, -1)
  local backPointers = self._backPointer:view(self._remaining, -1)
  for _ = 1, nBest do
    local hypothesis, finished
    if prevId <= #prevCompleted and prevCompleted[prevId][2] > normScores[remainingId][currId] then
      hypothesis = prevCompleted[prevId]
      finished = true
      prevId = prevId + 1
    else
      finished = (completed[remainingId][currId] == 1)
      if finished then
        local normScore = normScores[remainingId][currId]
        local token = tokens[remainingId][currId]
        local backPointer = backPointers[remainingId][currId]
        hypothesis = {origId, normScore, token, backPointer, self._step}
      end
      currId = currId + 1
    end
    table.insert(hypotheses, {hypothesis = hypothesis, finished = finished})
  end
  return hypotheses
end
return Beam
