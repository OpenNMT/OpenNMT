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

local function beamSelect(v, selIndexes)
  return onmt.utils.Tensor.recursiveApply(v, function (h)
    local batchSize = selIndexes:size(1)
    local beamSize = selIndexes:size(2)
    return h:index(1, selIndexes:view(-1):long()
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

local function selectBatch(v, remaining)
  return onmt.utils.Tensor.recursiveApply(v, function (h)
    if not torch.isTensor(remaining) then
      remaining = torch.LongTensor(remaining)
    end
    return h:index(1, remaining)
  end)
end

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

  * `stepFunction` - this function specifies how to go one step forward. It will take a table `stepInputs` as input, produce a table `stepOutputs` as output. All tensors inside tables must have the same first dimension `batchSize`. `stepOutputs[1]` should be of shape (`batchSize`, `numTokens`), which denotes the scores in this step.
  * `feedFunction` - this function specifies how to prepare the input `stepInputs` to stepFunction given `stepOutputs` and predictions from last time step. All tensors inside tables must have the same first dimension `batchSize`. Input: `topIndexes`, a tensor of shape (`batchSize`), which are the predictions from last time step; `stepOutputs`, the corresponding output of `stepFunction` from last time step. At the first time step, both `stepOutputs` and `topIndexes` will be passed as nil, and the returned values of this function acts as an initializer.
  * `maxSeqLength` - maximum output sequence length.
  * `endSymbol` - end of sequence symbol. [onmt.Constants.EOS]

]]
function BeamSearcher:__init(stepFunction, feedFunction, maxSeqLength,
    keptIndexes, endSymbol)
  self.stepFunction = stepFunction
  self.feedFunction = feedFunction
  self.maxSeqLength = maxSeqLength
  self.endSymbol = endSymbol or onmt.Constants.EOS
  self.keptIndexes = keptIndexes
end

--[[ Perform beam search.

Parameters:

  * `beamSize` - this function specifies how to go one step forward. It will take a table `stepInputs` as input, produce a table `stepOutputs` as output. All tensors inside tables must have the same first dimension `batchSize`. `stepOutputs[1]` should be of shape (`batchSize`, `numTokens`), which denotes the scores in this step. [1]
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
  local topIndexes -- kept top beamSize ids in the beam, (batchSize, beamSize)
  local beamScores -- scores in the beam, (batchSize, beamSize)
  local stepOutputs
  local remainingBatchIdToOrigBatchId = {}

  local t = 1
  while t <= self.maxSeqLength do
    local flatTopIndexes = topIndexes
    if flatTopIndexes then
      flatTopIndexes = flatTopIndexes:view(-1)
    end
    local nextInputs = self.feedFunction(stepOutputs, flatTopIndexes) -- prepare next step inputs (when t == 1, initialize)
    stepOutputs = self.stepFunction(nextInputs) -- go one step forward
    local scores = stepOutputs[1] -- if t == 1, (origBatchSize, vocabSize); else (remainingBatchSize * beamSize, vocabSize)
    if vocabSize then
      assert (vocabSize == scores:size(2))
    else
      vocabSize = scores:size(2)
    end
    -- figure out the top k indexes, and where they come from
    local rawIndexes, remainingBatchSize
    if t == 1 then
      self.origBatchSize = scores:size(1)
      remainingBatchSize = self.origBatchSize -- completed sequences will be removed from batch, so batch size changes
      for b = 1, self.origBatchSize do
        remainingBatchIdToOrigBatchId[b] = b
      end
      beamScores, rawIndexes = scores:topk(self.beamSize, 2, true, true)
      rawIndexes:add(-1)
      topIndexes = onmt.utils.Cuda.convert(rawIndexes:double()) + 1 -- (origBatchSize, beamSize)
    else
      remainingBatchSize = math.floor(scores:size(1) / self.beamSize)
      if self.nBest > 1 then
        local minScore = -9e9
        scores:add(onmt.utils.Cuda.convert(topIndexes:view(-1):eq(self.endSymbol):double()):mul(minScore):view(-1, 1):expand(scores:size(1), vocabSize)) -- once EOS encountered, set other token scores to -inf
      end
      scores:select(2, self.endSymbol):maskedFill(topIndexes:view(-1):eq(self.endSymbol), 0) -- once EOS encountered, stuck at that point
      local totalScores = (scores:view(remainingBatchSize, self.beamSize, vocabSize) + beamScores:view(remainingBatchSize, self.beamSize, 1):expand(remainingBatchSize, self.beamSize, vocabSize)):view(remainingBatchSize, self.beamSize * vocabSize) -- (remainingBatchSize, beamSize * vocabSize)
      beamScores, rawIndexes = totalScores:topk(self.beamSize, 2, true, true) -- (remainingBatchSize, beamSize)
      rawIndexes = onmt.utils.Cuda.convert(rawIndexes:double())
      rawIndexes:add(-1)
      topIndexes = onmt.utils.Cuda.convert(rawIndexes:double():fmod(vocabSize)) + 1 -- (remainingBatchSize, beamSize)
    end
    local beamParents = onmt.utils.Cuda.convert(rawIndexes:int()/vocabSize + 1) -- (remainingBatchSize, beamSize)
    -- use the top k indexes to select the stepOutputs
    if t == 1 then -- beamReplicate
      stepOutputs = beamReplicate(stepOutputs, self.beamSize) -- convert to (origBatchSize * beamSize, *)
    end
    stepOutputs = beamSelect(stepOutputs, beamParents) -- (remainingBatchSize * beamSize, *)

    -- judge whether end has been reached use topIndexes (batchSize, beamSize)
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
    table.insert(self.beamParentsHistory, beamParents) -- (remainingBatchSize, beamSize)
    table.insert(self.beamScoresHistory, beamScores:clone()) -- (remainingBatchSize, beamSize)
    local keptStepOutputs = {}--stepOutputs
    for _, val in pairs(self.keptIndexes) do
      keptStepOutputs[val] = stepOutputs[val]
    end
    table.insert(self.stepOutputsHistory, onmt.utils.Tensor.recursiveClone(keptStepOutputs)) -- newRemainingBatchSize
    if newRemainingBatchSize ~= remainingBatchSize then -- some batches are finished
      if #remaining ~= 0 then
        stepOutputs = rcToFlat(selectBatch(flatToRc(stepOutputs, self.beamSize), remaining))
        topIndexes = selectBatch(topIndexes, remaining)
        beamScores = selectBatch(beamScores, remaining)
      else
        break
      end
    end
    table.insert(self.topIndexesHistory, topIndexes:clone()) -- newRemainingBatchSize
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

  -- final decoding
  for b = 1, self.origBatchSize do
    predictions[b] = {}
    outputs[b] = {}
    local t = self.completedTimeStep[b]
    local parentIndex, topIndex
    parentIndex = k
    if t then
      scores[b] = self.beamScoresHistory[t][self.origBatchIdToRemainingBatchId[t - 1][b]][parentIndex]
      while t > 1 do
        outputs[b][t] = selectBatchBeam(self.stepOutputsHistory[t], self.beamSize, self.origBatchIdToRemainingBatchId[t - 1][b], parentIndex)
        parentIndex = self.beamParentsHistory[t][self.origBatchIdToRemainingBatchId[t - 1][b]][parentIndex]
        topIndex = self.topIndexesHistory[t - 1][self.origBatchIdToRemainingBatchId[t - 1][b]][parentIndex]
        predictions[b][t - 1] = topIndex
        t = t - 1
      end
      outputs[b][1] = selectBatchBeam(self.stepOutputsHistory[1], self.beamSize, b, parentIndex)
    else
      t = self.maxSeqLength
      scores[b] = self.beamScoresHistory[t][self.origBatchIdToRemainingBatchId[t - 1][b]][parentIndex]
      topIndex = self.topIndexesHistory[t][self.origBatchIdToRemainingBatchId[t][b]][parentIndex] -- 1 ~ beamSize
      predictions[b][t] = topIndex
      while t > 1 do
        outputs[b][t] = selectBatchBeam(self.stepOutputsHistory[t], self.beamSize, self.origBatchIdToRemainingBatchId[t - 1][b], parentIndex)
        parentIndex = self.beamParentsHistory[t][self.origBatchIdToRemainingBatchId[t - 1][b]][parentIndex]
        topIndex = self.topIndexesHistory[t - 1][self.origBatchIdToRemainingBatchId[t - 1][b]][parentIndex] -- 1 ~ beamSize
        outputs[b][t - 1] = selectBatchBeam(self.stepOutputsHistory[t - 1], self.beamSize, self.origBatchIdToRemainingBatchId[t - 1][b], parentIndex)
        predictions[b][t - 1] = topIndex
        t = t - 1
      end
      outputs[b][1] = selectBatchBeam(self.stepOutputsHistory[1], self.beamSize, b, parentIndex)
    end
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
