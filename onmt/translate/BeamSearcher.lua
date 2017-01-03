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


--[[Helper function.

Parameters:

  * `v` - tensor
  * `x` - reference tensor

Returns: if `x` is cuda tensor, return `v:cuda()`; otherwise, return `v`.

--]]
local function localize(v, x)
  if x:type() == 'torch.CudaTensor' then
    return v:cuda()
  else
    return v
  end
end

--[[Constructor

Parameters:

  * `advancer` - an `onmt.translate.BeamSearchAdvancer` object.
  * `endSymbol` - end symbol in the vocabulary. [onmt.Constants.EOS]

]]
function BeamSearcher:__init(advancer, endSymbol)
  self.stepFunction = advancer.step
  self.keptStateIndexes = advancer.keptStateIndexes
  self.endSymbol = endSymbol or onmt.Constants.EOS
end

--[[ Perform beam search.

Parameters:

  * `beamSize` - beam size. [1]
  * `nBest` - the `nBest` top hypotheses can be returned after performing beam search. This value must be smaller than or equal to `beamSize`. [1]

]]
function BeamSearcher:search(beamSize, nBest)
  assert (nBest <= beamSize)
  self.nBest = nBest or 1
  self.beamSize = beamSize or 1
  self.origBatchIdToRemainingBatchId = {}
  self.origBatchSize = nil
  self.topIndexesHistory = {}
  self.stepOutputsHistory = {}
  self.beamParentsHistory = {}
  self.beamScoresHistory = {}
  self.completedTimeStep = {}

  local vocabSize
  local topIndexes
  local beamScores
  local stepOutputs, scores
  local remainingBatchIdToOrigBatchId = {}

  local t = 1
  while true do
    local flatTopIndexes = topIndexes
    if flatTopIndexes then
      flatTopIndexes = flatTopIndexes:view(-1)
    end
    -- Go one step forward
    scores, stepOutputs = self.stepFunction(stepOutputs, flatTopIndexes)
    if scores == nil then
      self.maxSeqLength = t
      break
    end
    if vocabSize then
      assert (vocabSize == scores:size(2))
    else
      vocabSize = scores:size(2)
    end
    -- Figure out the top k indexes, and which beam do they come from
    local rawIndexes, remainingBatchSize
    if t == 1 then
      self.origBatchSize = scores:size(1)
      remainingBatchSize = self.origBatchSize
      for b = 1, self.origBatchSize do
        remainingBatchIdToOrigBatchId[b] = b
      end
      beamScores, rawIndexes = scores:topk(self.beamSize, 2, true, true)
      rawIndexes:add(-1)
      topIndexes = localize(rawIndexes:double(), rawIndexes) + 1
    else
      remainingBatchSize = math.floor(scores:size(1) / self.beamSize)
      -- Set other tokens scores to -inf to avoid ABCD<EOS>FG on beam
      if self.nBest > 1 then
        local minScore = -9e9
        scores:add(localize(topIndexes:view(-1):eq(self.endSymbol):double(), scores)
          :mul(minScore):view(-1, 1):expand(scores:size(1), vocabSize))
      end
      -- Ensure that tokens after <EOS> remain <EOS> and scores do not change
      scores:select(2, self.endSymbol)
        :maskedFill(topIndexes:view(-1):eq(self.endSymbol), 0)
      local totalScores = (scores:view(remainingBatchSize, self.beamSize
        , vocabSize)
        + beamScores:view(remainingBatchSize, self.beamSize, 1)
        :expand(remainingBatchSize, self.beamSize, vocabSize))
        :view(remainingBatchSize, self.beamSize * vocabSize)
      beamScores, rawIndexes = totalScores:topk(self.beamSize, 2, true, true)
      rawIndexes = localize(rawIndexes:double(), rawIndexes)
      rawIndexes:add(-1)
      topIndexes = localize(rawIndexes:double():fmod(vocabSize), rawIndexes) + 1
    end
    local beamParents = localize(rawIndexes:int() / vocabSize + 1)
    -- Use the top k indexes to select the stepOutputs
    if t == 1 then
      -- Replicate batchSize hypotheses to batchSize * beamSize hypotheses
      stepOutputs = beamReplicate(stepOutputs, self.beamSize)
    end
    -- Select the on-beam states using the pointers
    stepOutputs = beamSelect(stepOutputs, beamParents)

    -- Judge whether end has been reached
    local remaining = {}
    local newRemainingBatchSize = 0
    self.origBatchIdToRemainingBatchId[t] = {}
    local remainingBatchIdToOrigBatchIdTemp = {}
    for b = 1, remainingBatchSize do
      local origBatchId = remainingBatchIdToOrigBatchId[b]
      local done = true
      for k = 1, nBest do
        if topIndexes[b][k] ~= self.endSymbol then
          done = false
        end
      end
      if not done then
        newRemainingBatchSize = newRemainingBatchSize + 1
        self.origBatchIdToRemainingBatchId[t][origBatchId] = newRemainingBatchSize
        remainingBatchIdToOrigBatchIdTemp[newRemainingBatchSize] = origBatchId
        table.insert(remaining, b)
      else
        self.completedTimeStep[origBatchId] = t
      end
    end
    remainingBatchIdToOrigBatchId = remainingBatchIdToOrigBatchIdTemp
    table.insert(self.beamParentsHistory, beamParents)
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
          flatToRc(stepOutputs, self.beamSize), remaining))
        topIndexes = selectBatch(topIndexes, remaining)
        beamScores = selectBatch(beamScores, remaining)
      else
        break
      end
    end
    table.insert(self.topIndexesHistory, topIndexes:clone())
    t = t + 1
  end
end

--[[ Get beam search predictions.

Parameters:

  * `k` - if set to k, then the k-th predictions will be returned. It must be smaller than or equal to `nBest` in `BeamSearcher:search`. [1]

Returns:

  * `predictions` - a table. predictions[b][t] stores the prediction in batch `b` and time step `t`.
  * `scores` - a table. scores[b] stores the prediction score of batch `b`.
  * `outputs` - a table. outputs[b][j][t] stores the j-th element in `stepOutputs` produced by `stepFunction` in batch `b` and time step `t`. In the case of an empty prediction, i.e. `predictions[b] == {}`, `outputs[b]` will be nil.

]]
function BeamSearcher:getPredictions(k)
  k = k or 1
  assert (k <= self.nBest)
  local predictions = {}
  local scores = {}
  local outputs = {}

  -- Decode
  for b = 1, self.origBatchSize do
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
      outputs[b][t] = selectBatchBeam(self.stepOutputsHistory[t], self.beamSize
        , self.origBatchIdToRemainingBatchId[t - 1][b], parentIndex)
      parentIndex = self.beamParentsHistory[t]
        [self.origBatchIdToRemainingBatchId[t - 1][b]][parentIndex]
      topIndex = self.topIndexesHistory[t - 1]
        [self.origBatchIdToRemainingBatchId[t - 1][b]][parentIndex]
      predictions[b][t - 1] = topIndex
      t = t - 1
    end
    outputs[b][1] = selectBatchBeam(self.stepOutputsHistory[1], self.beamSize
      , b, parentIndex)
    -- trim trailing EOS
    for s = #predictions[b], 1, -1 do
      if predictions[b][s] == self.endSymbol then
        predictions[b][s] = nil
        outputs[b][s + 1] = nil
      else
        break
      end
    end
  end
  -- transpose outputs
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

return BeamSearcher
