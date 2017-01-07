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
local function beamReplicate(v, beamSize)
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
local function beamSelect(v, indexes)
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
end

--[[ Perform beam search.

Parameters:

  * `beamSize` - beam size. [1]
  * `nBest` - the `nBest` top hypotheses can be returned after performing beam search. This value must be smaller than or equal to `beamSize`. [1]

Returns:

  * `predictions` - a table. predictions[b][t] stores the prediction in batch `b` and time step `t`.
  * `scores` - a table. scores[b] stores the prediction score of batch `b`.
  * `outputs` - a table. outputs[b][j][t] stores the j-th element in `stepOutputs` produced by `stepFunction` in batch `b` and time step `t`. In the case of an empty prediction, i.e. `predictions[b] == {}`, `outputs[b]` will be nil.

]]
function BeamSearcher:search(beamSize, nBest)
  assert (nBest <= beamSize)
  self.nBest = nBest or 1
  beamSize = beamSize or 1

  -- Step stats
  self.history = { extensions = {},
                   states = {},
                   backPointers = {},
                   totalScores = {},
                   isComplete = {},
                   orig2Remaining = {}
                   remaining2Orig = {}
                 }
  -- Other stats
  self.stats = { extensionSize = nil,
                 batchSize = nil,
                 completedStep = {},
               }

  -- Initialize
  local extensions, states = self.advancer.init()

  local t = 1
  local hypotheses = {}
  local scores, kMaxScores, kMaxIds, remaining, prevComplete
  while remaining == nil or remaining > 0 do
    -- Forward
    states = self.advancer:forward(extensions, states)
    -- Expand
    scores = self.advancer:expand(states)
    self.stats.batchSize = self.stats.batchSize or scores:size(1)
    self.stats.extensionSize = self.stats.extensionSize or scores:size(2)
    remaining = remaining or scores:size(1)
    -- Select extensions with k-max scores (and satisfying filters)
    kMaxScores, kMaxIds, backPointers = self:_kArgMax(beamSize,
                                                      kMaxScores, 
                                                      scores, t,
                                                      self.advancer.filter
                                                     )
    -- Update hypotheses
    hypotheses = self:_updateHyps(hypotheses, backPointers, kMaxIds)
    -- Index states using backpointers to feed next step
    states = self:_indexStates(states, backPointers, t, beamSize)
    -- Determine which states are complete
    local complete = self.advancer:isComplete(hypotheses, states)
    if prevComplete then
      complete = (complete + beamSelect(prevComplete, backPointers)):ge(1)
    end
    prevComplete = complete
    -- Keep track of history
    self:_updateHistory(backPointers, states, complete)
    -- Remove complete batches
    local remainingIds = self:_removeComplete(complete:viewAs(kMaxIds),
                                              states, hypotheses, t)
    t = t + 1
  end
  -- Return predictions
  local results = {}
  for k = 1, nBest do
    local predictions, scores, states = self:_getPredictions(k)
    results[k] = { hypotheses = hypotheses, scores = scores, states = states}
  end
  return resultss
end

function BeamSearcher:_getPredictions(k)
  local predictions = {}
  local scores = {}
  local outputs = {}

  -- Decode
  for b = 1, self.stats.batchSize do
    predictions[b] = {}
    outputs[b] = {}
    local t = self.completedTimeStep[b]
    local parentIndex, topIndex
    parentIndex = k
    if t == nil then
      t = self.maxSeqLength
      scores[b] = self.beamScoresHistory[t]
        [self.origBatchIdToRemainingBatchId[t - 1][b]][parentIndex]
      topIndex = self.topIndexesHistory[t]
        [self.origBatchIdToRemainingBatchId[t][b]][parentIndex]
      predictions[b][t] = topIndex
    end
    scores[b] = self.beamScoresHistory[t]
      [self.origBatchIdToRemainingBatchId[t - 1][b]][parentIndex]
    while t > 1 do
      outputs[b][t] = selectBatchBeam(self.stepOutputsHistory[t], beamSize
        , self.origBatchIdToRemainingBatchId[t - 1][b], parentIndex)
      parentIndex = self.backPointersHistory[t]
        [self.origBatchIdToRemainingBatchId[t - 1][b]][parentIndex]
      topIndex = self.topIndexesHistory[t - 1]
        [self.origBatchIdToRemainingBatchId[t - 1][b]][parentIndex]
      predictions[b][t - 1] = topIndex
      t = t - 1
    end
    outputs[b][1] = selectBatchBeam(self.stepOutputsHistory[1], beamSize
      , b, parentIndex)
    -- Trim trailing EOS
    for s = #predictions[b], 1, -1 do
      if predictions[b][s] == self.endSymbol then
        predictions[b][s] = nil
        outputs[b][s + 1] = nil
      else
        break
      end
    end
  end

  -- Transpose outputs
  local outputsTemp = {}
  for b = 1, #outputs do
    outputsTemp[b] = {}
    for t = 1, #outputs[b] do
      for j, _ in pairs(outputs[b][t]) do
        outputsTemp[b][j] = outputsTemp[b][j] or {}
        outputsTemp[b][j][t] = outputs[b][t][j]
      end
    end
  end
  outputs = outputsTemp
  return predictions, scores, outputs
end

-- Find the top k extensions (satisfying filters)
function BeamSearcher:_kArgMax(beamSize, prevKMaxScores, scores, t, filterFunction, hypotheses)
  local kMaxScores, kMaxIds, backPointers
  local loop = 0
  local filtersSatisfied = false
  while not filtersSatisfied do
    loop = loop + 1
    if t == 1 then
      kMaxScores, kMaxIds = scores:topk(beamSize, 2, true, true)
      backPointers = kMaxIds.new():resizeAs(kMaxIds):fill(1)
    else
      local extensionSize = scores:size(2)
      remaining = math.floor(scores:size(1) / beamSize)
      local complete = self.history.isComplete[#self.history.isComplete]
      -- Set other tokens scores to -inf to avoid ABCD<EOS>FG being on beam
      if self.nBest > 1 then
        local maskScores = scores.new():resize(scores:size(1)):fill(0)
        maskScores:maskedFill(complete, -math.huge)
        scores:add(maskScores:view(-1, 1):expandAs(scores))
      end
      -- Ensure that complete hypotheses remain and their scores do not change
      scores:select(2, 1):maskedFill(complete, 0)
      local totalScores = (scores:view(remaining, beamSize, -1)
                           + prevKMaxScores:view(remaining, beamSize, 1)
                                           :expand(remaining, beamSize, extensionSize)
                          ):view(remaining, -1)
      kMaxScores, kMaxIds = totalScores:topk(beamSize, 2, true, true)
      kMaxIds:add(-1)
      backPointers = (kMaxIds / extensionSize):add(1)
      kMaxIds = kMaxIds:fmod(extensionSize) + 1
    end
    if not filterFunction then
      break
    end
    -- Prune hypotheses if necessary
    assert (loop <= scores:size(2), 'All hypotheses do not satisfy filters!')
    local newHypotheses = self:_updateHyps(hypotheses, backPointers, kMaxIds)
    local newStates = self:_indexStates(states, backPointers, t, beamSize)
    local unSatisfied = filterFunction(newHypotheses, newStates):eq(1)
    if not unSatisfied:any() then
      filtersSatisfied = true
    else
      scores:view(-1):maskedFill(unSatisfied, -math.huge)
    end
  end
  return kMaxScores, kMaxIds, backPointers
end

function BeamSearcher:_indexStates(states, backPointers, t, beamSize)
  if t == 1 then
    -- Replicate batchSize hypotheses to batchSize * beamSize hypotheses
    states = beamReplicate(states, beamSize)
  end
  -- Select the on-beam states using the pointers
  states = beamSelect(states, backPointers)
  return states
end

function BeamSearcher:_updateHyps(hypotheses, backPointers, kMaxIds)
  hypotheses = beamSelect(hypotheses, backPointers)
  hypotheses[#hypotheses] = kMaxIds:clone():view(-1)
  return hypotheses
end

function BeamSearcher:_removeComplete(complete, states, hypotheses, t)
  local prevComplete = self.history.isComplete[t - 1]
  if isComplete then
    complete = (complete + prevComplete):ge(1)
  end
  complete = complete:view(self.stats.batchSize, -1)
  local remainingIds = {}
  self.history.orig2Remaining[t] = {}
  self.history.remaining2Orig[t] = {}
  local remaining = 0
  for b = 1, self.stats.batchSize do
    local orig = self.history.remaining2Orig[t - 1][b] or b
    local done = true
    for k = 1, nBest do
      if complete[b][k] ~= 0 then
        done = false
      end
    end
    if not done then
      remaining = remaining + 1
      self.history.orig2Remaining[t][orig] = remaining
      self.history.remaining2Orig[t][remaining] = orig
      table.insert(remainingIds, b)
    else
      self.stats.completedStep[orig] = t
    end
  end
  return remainingIds
end

function BeamSearcher:_keepTrack(complete)
  table.insert(self.backPointersHistory, backPointers)
  table.insert(self.beamScoresHistory, beamScores:clone())
  local keptStepOutputs = {}
  for _, val in pairs(self.keptStateIndexes) do
    keptStepOutputs[val] = stepOutputs[val]
  end
  table.insert(self.stepOutputsHistory,
    onmt.utils.Tensor.recursiveClone(keptStepOutputs))
  -- Remove finished batches
  if newRemainingBatchSize ~= remainingBatchSize then
    if #remaining ~= 0 then
      stepOutputs = rcToFlat(selectBatch(
        flatToRc(stepOutputs, beamSize), remaining))
      topIndexes = selectBatch(topIndexes, remaining)
      beamScores = selectBatch(beamScores, remaining)
    else
      break
    end
  end
  table.insert(self.topIndexesHistory, topIndexes:clone())
end
return BeamSearcher
