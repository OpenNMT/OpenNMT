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
  * `prevFilterFactor` - optional, set this only if filter is being used. Before applying filters, hypotheses with top `beamSize * preFilterFactor` scores will be considered; afterwards, the hypotheses that do not satisfy filters are pruned, and then the on-beam top `beamSize` hypotheses will be selected from the remaining ones. If the returned hypotheses voilate filters, then consider setting this to a larger value. [1]

Returns: a table `results`. `results[n]` contains the n-th best `hypotheses`, `scores` and `states`. `hypotheses[b][t]` stores the hypothesis in batch `b` and step `t`. `scores[b]` stores the hypothesis score of batch `b`. `states[b][j][t]` stores the j-th element in `states` in batch `b` and step `t`.

]]
function BeamSearcher:search(beamSize, nBest, prevFilterFactor)
  self.nBest = nBest or 1
  self.beamSize = beamSize or 1
  self.prevFilterFactor = prevFilterFactor or 1

  local beams = {}
  local finished = {}

  -- Initialize the beam.
  beams[1] = self.advancer.init()
  local remaining = beams[1]:remaining()
  if beams[1]:tokens():size(1) ~= remaining * beamSize then
    beams[1]:replicate()
  end
  local t = 1
  while remaining > 0 do
    -- Update beam states based on new tokens.
    self.advancer:update(beams[t])

    -- Expand beams by all possible tokens and return the scores.
    local scores = self.advancer:expand(beams[t])
    
    -- Find k best next beams (maintained by BeamSearcher).
    self:_findKBest(beams, scores)

    -- Determine which hypotheses are complete.
    local completed = self.advancer:isComplete(beams[t])

    -- Remove completed hypotheses (maintained by BeamSearcher).
    local batchIds, hypotheses = self:_completeHypotheses(beams, completed)

    for b = 1, #batchIds do
      finished[batchIds[b]] = hypotheses[b]
    end

    t = t + 1
    remaining = beams[t]:remaining()
  end
  return finished
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
      hypotheses[b][t] = self.history.tokens[t][remaining][fromBeam]
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

-- Find the top beamSize hypotheses (satisfying filters)
function BeamSearcher:_findKBest(beams, scores)
  local t = #beams
  local vocabSize = scores:size(2)
  local expandedScores = beam:expandScores(scores)

  -- Find top beamSize * prevFilterFactor hypotheses
  local considered = self.beamSize * self.prevFilterFactor
  local consideredScores, consideredIds = expandedScores:topk(considered, 2,
                                                              true, true)
  consideredIds:add(-1)
  local consideredBackPointers = (consideredIds:clone():div(vocabSize)):add(1)
  local consideredToken = consideredIds:fmod(vocabSize):add(1):view(-1)

  local newBeam = beams[t]:nextBeam(consideredToken, consideredScores,
                                    consideredBackPointers, self.beamSize)

  -- Prune hypotheses if necessary
  local prune = self.advancer:filter(newBeam)
  if prune then
    consideredScores:view(-1):maskedFill(prune, -math.huge)
  end

  -- Find top beamSize hypotheses
  local kBestScores, kBestIds, backPointers
  if ( (not prune) or (not prune:any()) ) and (self.prevFilterFactor == 1) then
    beams[t + 1] = newBeam
  else
    local kBestScores, kBestIds = consideredScores:topk(self.beamSize, 2,
                                                        true, true)
    local backPointers = consideredBackPointers:gather(2, kBestIds)
    local token = consideredToken:gather(2, kBestIds)
    local newBeam = beams[t]:nextBeam(token, kBestScores,
                                      backPointers, self.beamSize)
    beams[t + 1] = newBeam
  end
end

function BeamSearcher:_completeHypotheses(beams, completed)
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
      tokens = selectBatch(tokens, remainingIds)
      totalScores = selectBatch(totalScores, remainingIds)
    end
  end
  return prevComplete, tokens, totalScores, states, hypotheses, remaining
end

function BeamSearcher:_trackHistory(totalScores, tokens, backPointers,
                                    states, complete)
  table.insert(self.history.totalScores, totalScores:clone())
  table.insert(self.history.tokens, tokens:clone())
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
