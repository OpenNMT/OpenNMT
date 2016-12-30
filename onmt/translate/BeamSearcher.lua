--[[ BeamSearcher is a class used for general beam search.
--]]
local BeamSearcher = torch.class('BeamSearcher')

local function beamReplicate(h, beamSize)
  local hOut
  if torch.type(h) == 'table' then
    hOut = {}
    for key, val in pairs(h) do
      hOut[key] = beamReplicate(val, beamSize)
    end
    return hOut
  end
  if torch.isTensor(h) then
    local batchSize = h:size(1)
    local sizes = {}
    for j = 2, #h:size() do
        sizes[j - 1] = h:size(j)
    end
    hOut = h:contiguous():view(batchSize, 1, table.unpack(sizes)):expand(batchSize, beamSize, table.unpack(sizes)):contiguous():view(batchSize * beamSize, table.unpack(sizes))
  else
    hOut = h
  end
  return hOut
end

local function beamSelect(h, selIndexes)
  local hOut
  if torch.type(h) == 'table' then
    hOut = {}
    for key, val in pairs(h) do
      hOut[key] = beamSelect(val, selIndexes)
    end
    return hOut
  end
  if torch.isTensor(h) then
    local batchSize = selIndexes:size(1)
    local beamSize = selIndexes:size(2)
    hOut = h:index(1, selIndexes:view(-1) + onmt.utils.Cuda.convert(torch.range(0, (batchSize - 1) * beamSize, beamSize):long()):contiguous():view(batchSize, 1):expand(batchSize, beamSize):contiguous():view(-1))
  else
    hOut = h
  end
  return hOut
end

-- maybe can use a more general wraper
local function flatToRc(h, beamSize)
  local hOut
  if torch.type(h) == 'table' then
    hOut = {}
    for key, val in pairs(h) do
      hOut[key] = flatToRc(val, beamSize)
    end
    return hOut
  end
  if torch.isTensor(h) then
    local batchSize = math.floor(h:size(1) / beamSize)
    local sizes = {}
    for j = 2, #h:size() do
        sizes[j - 1] = h:size(j)
    end
    hOut = h:view(batchSize, beamSize, table.unpack(sizes))
  else
    hOut = h
  end
  return hOut
end

local function selectBatch(h, remaining)
  if torch.type(remaining) == 'table' then
    remaining = onmt.utils.Cuda.convert(torch.Tensor(remaining))
  end
  local hOut
  if torch.type(h) == 'table' then
    hOut = {}
    for key, val in pairs(h) do
      hOut[key] = selectBatch(val, remaining)
    end
    return hOut
  end
  if torch.isTensor(h) then
    hOut = h:index(1, remaining)
  else
    hOut = h
  end
  return hOut
end

local function selectBatchBeam(h, beamSize, batch, beam)
  if torch.type(remaining) == 'table' then
    remaining = onmt.utils.Cuda.convert(torch.Tensor(remaining))
  end
  local hOut
  if torch.type(h) == 'table' then
    hOut = {}
    for key, val in pairs(h) do
      hOut[key] = selectBatch(val, beamSize, batch, beam)
    end
    return hOut
  end
  if torch.isTensor(h) then
    local batchSize = math.floor(h:size(1) / beamSize)
    local sizes = {}
    for j = 2, #h:size() do
        sizes[j - 1] = h:size(j)
    end
    hOut = h:view(batchSize, beamSize, table.unpack(sizes))
    hOut = h[{batch, beam}]
  else
    hOut = h
  end
  return hOut
end

local function rcToFlat(h)
  local hOut
  if torch.type(h) == 'table' then
    hOut = {}
    for key, val in pairs(h) do
      hOut[key] = rcToFlat(val)
    end
    return hOut
  end
  if torch.isTensor(h) then
    local sizes = {}
    sizes[1] = h:size(1) * h:size(2)
    for j = 3, #h:size() do
        sizes[j - 1] = h:size(j)
    end
    hOut = h:view(table.unpack(sizes))
  else
    hOut = h
  end
  return hOut
end

function BeamSearcher:__init()
end

function BeamSearcher:search(stepFunction, feedFunction, beamSize, maxSeqLength, endSymbol, nBest, maxScore)
  endSymbol = endSymbol or onmt.Constants.EOS
  nBest = nBest or 1
  maxScore = maxScore or 0.0
  local stepOutputs
  local topIndexes -- kept top beamSize ids in the beam, (batchSize, beamSize)
  local beamScores -- scores in the beam, (batchSize, beamSize)
  local vocabSize
  local origBatchSize
  local origBatchIdToRemainingBatchId, remainingBatchIdToOrigBatchId = {}, {}
  local completed = {}
  local beamParentsHistory = {}
  local topIndexesHistory = {}
  local stepOutputsHistory = {}
  local beamScoresHistory = {}
  local t = 1
  while t <= maxSeqLength do
    local nextInputs = feedFunction(stepOutputs, topIndexes)
    stepOutputs = stepFunction(nextInputs)
    local scores = stepOutputs[1] -- if t == 1, (origBatchSize, vocabSize); else (remainingBatchSize * beamSize, vocabSize)
    if vocabSize == nil then
      vocabSize = scores:size(2)
    else
      assert (vocabSize == scores:size(2))
    end
    -- figure out the top k indexes, and where they come from
    local rawIndexes, remainingBatchSize
    if t == 1 then
      origBatchSize = scores:size(1)
      remainingBatchSize = origBatchSize
      for b = 1, origBatchSize do
        remainingBatchIdToOrigBatchId[b] = b
      end
      beamScores, rawIndexes = scores:topk(beamSize, 2, true, true)
      rawIndexes:add(-1)
      topIndexes = onmt.utils.Cuda.convert(rawIndexes:double()) + 1 -- (origBatchSize, beamSize)
    else
      remainingBatchSize = math.floor(scores:size(1) / beamSize)
      scores:select(2, endSymbol):maskedFill(topIndexes:view(-1):eq(endSymbol), maxScore) -- once padding or EOS encountered, stuck at that point
      local totalScores = (scores:view(remainingBatchSize, beamSize, vocabSize) + beamScores:view(remainingBatchSize, beamSize, 1):expand(remainingBatchSize, beamSize, vocabSize)):view(remainingBatchSize, beamSize * vocabSize) -- (remainingBatchSize, beamSize * vocabSize)
      beamScores, rawIndexes = totalScores:topk(beamSize, 2, true, true) -- (remainingBatchSize, beamSize)
      rawIndexes = onmt.utils.Cuda.convert(rawIndexes:double())
      rawIndexes:add(-1)
      topIndexes = onmt.utils.Cuda.convert(rawIndexes:double():fmod(vocabSize)) + 1 -- (remainingBatchSize, beamSize)
    end
    local beamParents = onmt.utils.Cuda.convert(rawIndexes:int()/vocabSize + 1) -- (remainingBatchSize, beamSize)
    -- use the top k indexes to select the stepOutputs
    if t == 1 then -- beamReplicate
      stepOutputs = beamReplicate(stepOutputs, beamSize) -- convert to (origBatchSize * beamSize, *)
    end
    stepOutputs = beamSelect(stepOutputs, beamParents) -- (remainingBatchSize * beamSize, *)

    -- judge whether end has been reached use topIndexes (batchSize, beamSize)
    local remaining = {}
    local newId = 0
    origBatchIdToRemainingBatchId[t] = {}
    local remainingBatchIdToOrigBatchIdTemp = {}
    for b = 1, remainingBatchSize do
      local origBatchId = remainingBatchIdToOrigBatchId[b]
      local done = true
      for k = 1, nBest do
        if topIndexes[b][k] ~= endSymbol then
          done = false
        end
      end
      if not done then
        newId = newId + 1
        origBatchIdToRemainingBatchId[t][origBatchId] = newId
        remainingBatchIdToOrigBatchIdTemp[newId] = origBatchId
        table.insert(remaining, b)
      else
        completed[origBatchId] = t
      end
    end
    remainingBatchIdToOrigBatchId = remainingBatchIdToOrigBatchIdTemp
    if newId ~= remainingBatchSize then
      if #remaining ~= 0 then
        stepOutputs = rcToFlat(selectBatch(flatToRc(stepOutputs, beamSize), remaining))
        topIndexes = selectBatch(topIndexes, remaining)
        beamScores = selectBatch(beamScores, remaining)
      else
        table.insert(beamParentsHistory, beamParents) -- remainingBatchSize
        table.insert(beamScoresHistory, beamScores) -- remainingBatchSize
        break
      end
    end
    table.insert(beamParentsHistory, beamParents) -- remainingBatchSize
    table.insert(beamScoresHistory, beamScores) -- remainingBatchSize
    table.insert(topIndexesHistory, topIndexes:clone()) -- newRemainingBatchSize
    table.insert(stepOutputsHistory, onmt.utils.recursiveClone(stepOutputs)) -- newRemainingBatchSize
    t = t + 1
  end

  self.origBatchSize = origBatchSize
  self.beamSize = beamSize
  self.beamParentsHistory = beamParentsHistory
  self.topIndexesHistory = topIndexesHistory
  self.stepOutputsHistory = stepOutputsHistory
  self.beamScoresHistory = beamScoresHistory
  self.origBatchIdToRemainingBatchId = origBatchIdToRemainingBatchId
  self.nBest = nBest
  self.endSymbol = endSymbol
end

function BeamSearcher:getPredictions(k)
  assert (k <= self.nBest)
  local predictions = {}
  local scores = {}
  local outputs = {}

  local origBatchSize = self.origBatchSize
  local beamParentsHistory = self.beamParentsHistory
  local beamScoresHistory = self.beamScoresHistory
  local topIndexesHistory = self.topIndexesHistory
  local stepOutputsHistory = self.stepOutputsHistory
  local origBatchIdToRemainingBatchId = self.origBatchIdToRemainingBatchId

  -- final decoding
  for b = 1, origBatchSize do
    predictions[b] = {}
    outputs[b] = {}
    t = completed[b]
    local parentIndex, topIndex
    parentIndex = k
    if t then
      scores[b] = beamParentsHistory[t][origBatchIdToRemainingBatchId[t-1][b]][parentIndex]
      while t > 1 do
        parentIndex = beamParentsHistory[t][origBatchIdToRemainingBatchId[t - 1][b]][parentIndex]
        topIndex = topIndexesHistory[t - 1][origBatchIdToRemainingBatchId[t - 1][b]][parentIndex] 
        outputs[b][t - 1] = selectBatchBeam(stepOutputsHistory[t - 1], self.beamSize, origBatchIdToRemainingBatchId[t - 1][b], parentIndex)
        predictions[b][t - 1] = topIndex
        t = t - 1
      end
    else
      t = maxSeqLength
      scores[b] = beamParentsHistory[t][origBatchIdToRemainingBatchId[t-1][b]][parentIndex]
      topIndex = topIndexesHistory[t][origBatchIdToRemainingBatchId[t][b]][parentIndex] -- 1 ~ beamSize
      outputs[b][t] = stepOutputsHistory[t][origBatchIdToRemainingBatchId[t][b]][parentIndex]
      predictions[b][t] = topIndex
      while t > 1 do
        parentIndex = beamParentsHistory[t][origBatchIdToRemainingBatchId[t-1][b]][parentIndex]
        topIndex = topIndexesHistory[t - 1][origBatchIdToRemainingBatchId[t-1][b]][parentIndex] -- 1 ~ beamSize
        outputs[b][t - 1] = selectBatchBeam(stepOutputsHistory[t - 1], self.beamSize, origBatchIdToRemainingBatchId[t - 1][b], parentIndex)
        predictions[b][t - 1] = topIndex
        t = t - 1
      end
    end
    if self.nBest > 1 then -- trim trailing EOS
      for t = #predictions[b], 1, -1 do
        if predictions[b][t] == self.endSymbol then
          predictions[b][t] = nil
          outputs[b][t] = nil
        else
          break
        end
      end
    end
  end
  return {predictions = predictions, scores = scores, outputs = outputs}
end
