--[[ Class for managing the internals of the beam search process.


      hyp1---hyp1---hyp1 -hyp1
          \             /
      hyp2 \-hyp2 /-hyp2--hyp2
                 /      \
      hyp3---hyp3---hyp3 -hyp3
      ========================

Takes care of beams, back pointers, and scores.
--]]
local BeamSearcher = torch.class('BeamSearcher')


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

--[[Helper function. Recursively index `(batchSize * beamSize, ...)`
  tensors by specified indexes.

Parameters:

  * `v` - tensor of size `(batchSize * beamSize, ...)` or a table containing such tensors
  * `indexes` - a tensor of size `(batchSize, beamSize)` specifying the indexes

Returns: Indexed `(batchSize * beamSize, ...)` tensor or a table containing such tensors

--]]
local function selectBeam(v, indexes)
  return onmt.utils.Tensor.recursiveApply(v, function (h)
    local batchSize = indexes:size(1)
    local beamSize = indexes:size(2)
    return h:index(1, indexes:view(-1):long()
                 + (torch.range(0, (batchSize - 1) * beamSize, beamSize):long())
                 :contiguous():view(batchSize, 1)
                 :expand(batchSize, beamSize):contiguous():view(-1))
  end)
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

--[[Constructor

Parameters:

  * `advancer` - an `onmt.translate.BeamSearchAdvancer` object.

]]
function BeamSearcher:__init(advancer)
  self.advancer = advancer
  -- Step stats
  self.history = { extensions = {},
                   states = {},
                   backPointers = {},
                   totalScores = {},
                   isComplete = {},
                   orig2Remaining = {},
                   remaining2Orig = {}
                 }
  -- Other stats
  self.stats = { extensionSize = nil,
                 batchSize = nil,
                 completedStep = {},
               }
end

--[[ Performs beam search.

Parameters:

  * `beamSize` - beam size. [1]
  * `nBest` - the `nBest` top hypotheses will be returned after beam search. [1]

Returns: a table `results`. `results[n]` contains the n-th best `hypotheses`, `scores` and `states`. `hypotheses[b][t]` stores the hypothesis in batch `b` and step `t`. `scores[b]` stores the hypothesis score of batch `b`. `states[b][j][t]` stores the j-th element in `states` in batch `b` and step `t`.

]]
function BeamSearcher:search(beamSize, nBest)
  nBest = nBest or 1
  beamSize = beamSize or 1
  assert (nBest <= beamSize)

  -- Initialize
  local extensions, states = self.advancer.init()

  local t = 1
  local hypotheses = {}
  local scores, totalScores, backPointers, remaining, prevComplete
  while remaining == nil or remaining > 0 do
    -- Forward
    states = self.advancer.forward(extensions:view(-1), states)
    -- Expand
    scores = self.advancer.expand(states)
    self.stats.batchSize = self.stats.batchSize or scores:size(1)
    self.stats.extensionSize = self.stats.extensionSize or scores:size(2)
    -- Select extensions with k-max scores (and satisfying filters)
    totalScores, extensions, backPointers, hypotheses, states
                                          = self:_kArgMax(beamSize, nBest,
                                                          totalScores, scores,
                                                          prevComplete,
                                                          self.advancer.filter,
                                                          hypotheses, states
                                                          )
    -- Determine which states are complete
    local complete = self.advancer.isComplete(hypotheses, states)
    if prevComplete then
      complete = (complete + selectBeam(prevComplete, backPointers)):ge(1)
    end
    prevComplete = complete
    -- Keep track of history
    self:_trackHistory(totalScores, extensions, backPointers, states, complete)
    -- Remove complete batches
    prevComplete, extensions, totalScores, states, hypotheses, remaining =
           self:_removeComplete(prevComplete, extensions, totalScores, states,
                                hypotheses, beamSize, nBest)
    t = t + 1
  end
  -- Return results
  local results = {}
  for k = 1, nBest do
    hypotheses, scores, states = self:_getResults(k, beamSize)
    results[k] = { hypotheses = hypotheses, scores = scores, states = states}
  end
  self.history, self.stats = nil, nil
  return results
end

function BeamSearcher:_getResults(k, beamSize)
  local hypotheses = {}
  local scores = {}
  local states = {}

  -- Decode
  for b = 1, self.stats.batchSize do
    hypotheses[b] = {}
    states[b] = {}
    local t = self.stats.completedStep[b]
    local fromBeam = k
    local remaining
    if t == 1 then
      remaining = b
    else
      remaining = self.history.orig2Remaining[t][b]
    end
    scores[b] = self.history.totalScores[t]
      [remaining][fromBeam]
    while t > 0 do
      if t == 1 then
        remaining = b
      else
        remaining = self.history.orig2Remaining[t][b]
      end
      states[b][t] = selectBatchBeam(self.history.states[t], beamSize,
                                     remaining, fromBeam)
      hypotheses[b][t] = self.history.extensions[t][remaining][fromBeam]
      local complete = self.history.isComplete[t]
      if selectBatchBeam(complete, beamSize, remaining, fromBeam) == 1 then
        states[b][t + 1] = nil
        hypotheses[b][t + 1] = nil
      end
      fromBeam = self.history.backPointers[t]
        [remaining][fromBeam]
      t = t - 1
    end
  end

  -- Transpose states
  local statesTemp = {}
  for b = 1, #states do
    statesTemp[b] = {}
    for t = 1, #states[b] do
      for j, _ in pairs(states[b][t]) do
        statesTemp[b][j] = statesTemp[b][j] or {}
        statesTemp[b][j][t] = states[b][t][j]
      end
    end
  end
  states = statesTemp
  return hypotheses, scores, states
end

-- Find the top k extensions (satisfying filters)
function BeamSearcher:_kArgMax(beamSize, nBest, totalScores, scores,
    prevComplete, filterFunction, hypotheses, states)
  local kMaxScores, kMaxIds, backPointers, newHypotheses, newStates
  local loop = 0
  local filtersSatisfied = false
  while not filtersSatisfied do
    loop = loop + 1
    local remaining
    if not totalScores then
      kMaxScores, kMaxIds = scores:topk(beamSize, 2, true, true)
      backPointers = kMaxIds.new():resizeAs(kMaxIds):fill(1)
      remaining = scores:size(1)
    else
      local extensionSize = scores:size(2)
      remaining = math.floor(scores:size(1) / beamSize)
      -- Set other tokens scores to -inf to avoid ABCD<EOS>FG being on beam
      if prevComplete then
        if nBest > 1 then
          local maskScores = scores.new():resize(scores:size(1)):fill(0)
          maskScores:maskedFill(prevComplete, -math.huge)
          scores:add(maskScores:view(-1, 1):expandAs(scores))
        end
        -- Ensure that complete hypotheses remain and their scores do not change
        scores:select(2, 1):maskedFill(prevComplete, 0)
      end
      local expandedScores = (scores:view(remaining, beamSize, -1)
                           + totalScores:view(remaining, beamSize, 1)
                                        :expand(remaining, beamSize, extensionSize)
                          ):view(remaining, -1)
      kMaxScores, kMaxIds = expandedScores:topk(beamSize, 2, true, true)
      kMaxIds:add(-1)
      backPointers = (kMaxIds:clone():div(extensionSize)):add(1)
      kMaxIds = kMaxIds:fmod(extensionSize):add(1)
    end
    newHypotheses = self:_updateHyps(hypotheses, backPointers, kMaxIds)
    newStates = self:_indexStates(states, backPointers,
                                        #hypotheses + 1, beamSize)
    -- Prune hypotheses if necessary
    assert (loop <= scores:size(2), 'All hypotheses do not satisfy filters!')
    local prune = filterFunction(newHypotheses, newStates)
    if not prune then
      break
    end
    if prevComplete then
      prune = (prune:eq(0):add(prevComplete)):eq(0)
    end
    if not prune:any() then
      filtersSatisfied = true
    else
      local pruneIds = prune:nonzero():view(-1)
      for b = 1, pruneIds:size(1) do
        local pruneId = pruneIds[b]
        local batchId = math.floor((pruneId - 1) / beamSize) + 1
        scores:view(remaining, -1, scores:size(2))[batchId]
                     [backPointers:view(-1)[pruneId]][kMaxIds:view(-1)[pruneId]]
                     = -math.huge
      end
    end
  end
  return kMaxScores, kMaxIds, backPointers, newHypotheses, newStates
end

function BeamSearcher:_indexStates(states, backPointers, t, beamSize)
  if t == 1 then
    -- Replicate batchSize hypotheses to batchSize * beamSize hypotheses
    states = replicateBeam(states, beamSize)
  end
  -- Select the on-beam states using the pointers
  states = selectBeam(states, backPointers)
  return states
end

function BeamSearcher:_updateHyps(hypotheses, backPointers, kMaxIds)
  hypotheses = selectBeam(hypotheses, backPointers)
  hypotheses[#hypotheses + 1] = kMaxIds:clone():view(-1)
  return hypotheses
end

function BeamSearcher:_removeComplete(prevComplete, extensions, totalScores, states,
    hypotheses, beamSize, nBest)
  local batchSize = math.floor(hypotheses[1]:size(1) / beamSize)
  local t = #hypotheses
  local complete = prevComplete:view(batchSize, -1)
  local remainingIds = {}
  self.history.orig2Remaining[t + 1] = {}
  self.history.remaining2Orig[t] = {}
  local remaining = 0
  for b = 1, batchSize do
    local orig
    if t == 1 then
      orig = b
    else
      orig = self.history.remaining2Orig[t - 1][b]
    end
    local done = true
    for k = 1, nBest do
      if complete[b][k] == 0 then
        done = false
      end
    end
    if not done then
      remaining = remaining + 1
      self.history.orig2Remaining[t + 1][orig] = remaining
      self.history.remaining2Orig[t][remaining] = orig
      table.insert(remainingIds, b)
    else
      self.stats.completedStep[orig] = t
    end
  end
  -- Remove finished batches
  if remaining < batchSize then
    if remaining > 0 then
      states = rcToFlat(selectBatch(
        flatToRc(states, beamSize), remainingIds))
      hypotheses = rcToFlat(selectBatch(
        flatToRc(hypotheses, beamSize), remainingIds))
      prevComplete = rcToFlat(selectBatch(
        flatToRc(prevComplete, beamSize), remainingIds))
      extensions = selectBatch(extensions, remainingIds)
      totalScores = selectBatch(totalScores, remainingIds)
    end
  end
  return prevComplete, extensions, totalScores, states, hypotheses, remaining
end

function BeamSearcher:_trackHistory(totalScores, extensions, backPointers,
                                    states, complete)
  table.insert(self.history.totalScores, totalScores:clone())
  table.insert(self.history.extensions, extensions:clone())
  table.insert(self.history.backPointers, backPointers:clone())
  table.insert(self.history.isComplete, complete:clone())
  local keptStates = {}
  local keptStateIndexes = self.advancer.keptStateIndexes or {}
  for _, val in pairs(keptStateIndexes) do
    keptStates[val] = states[val]
  end
  table.insert(self.history.states, onmt.utils.Tensor.recursiveClone(keptStates))
end
return BeamSearcher
