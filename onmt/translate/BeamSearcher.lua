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

--[[Constructor

Parameters:

  * `advancer` - an `onmt.translate.BeamSearchAdvancer` object.

]]
function BeamSearcher:__init(advancer)
  self.advancer = advancer
end

--[[ Performs beam search.

Parameters:

  * `beamSize` - beam size. [1]
  * `nBest` - the `nBest` top hypotheses will be returned after beam search. [1]
  * `beforeFilterFactor` - optional, set this only if filter is being used. Before applying filters, hypotheses with top `beamSize * preFilterFactor` scores will be considered. If the returned hypotheses voilate filters, then set this to a larger value to consider more. [1]

Returns: a table `results`. `results[n]` contains the n-th best `hypotheses`, `scores` and `states`. `hypotheses[b][t]` stores the hypothesis in batch `b` and step `t`. `scores[b]` stores the hypothesis score of batch `b`. `states[b][j][t]` stores the j-th element in `states` in batch `b` and step `t`.

]]
function BeamSearcher:search(beamSize, nBest, beforeFilterFactor)
  self.nBest = nBest or 1
  self.beamSize = beamSize or 1
  self.beforeFilterFactor = beforeFilterFactor or 1

  local beams = {}
  local finished = {}

  -- Initialize the beam.
  beams[1] = self.advancer.initBeam()
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
    local finishedBatches, finishedHypotheses = self:_completeHypotheses(beams, completed)

    for b = 1, #finishedBatches do
      finished[finishedBatches[b]] = finishedHypotheses[b]
    end
    t = t + 1
    remaining = beams[t]:remaining()
  end
  return finished
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

  -- Find top beamSize * beforeFilterFactor hypotheses
  local considered = self.beamSize * self.beforeFilterFactor
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
  if ( (not prune) or (not prune:any()) ) and (self.beforeFilterFactor == 1) then
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

    --local finishedBatches, hypotheses = self:_completeHypotheses(beams, completed)
function BeamSearcher:_completeHypotheses(beams, completed)
  local t = #beams
  local batchSize = math.floor(beams[t]:remaining() / self.beamSize)
  completed = completed:view(batchSize, -1)
  local scores = beams[t]:scores()

  local remainingId = 0
  local remainingIds = {}
  for b = 1, batchSize do
    local prevCompleted = beams[t]:completed(b)
    local origId = b
    if t > 1 then
      origId = beams[t - 1]:remaining2Orig(b)
    end
    local batchFinished = true
    local prevId = 1
    local currId = 1
    for k = 1, nBest do
      if prevId <= #prevCompleted then
        local prevScore = prevCompleted[prevId][1]
        if prevScore > scores[b][currId] then
          prevId = prevId + 1
        else
          if completed[b][currId] == 0 then
            batchFinished = false
            break
          end
          currId = currId + 1
        end
      end
    end
    if not batchFinished then
      remainingId = remainingId + 1
      beams[t]:setOrig2Remaining(origId, remainingId)
      beams[t]:setRemaining2Orig(remainingId, origId)
      table.insert(remainingIds, b)
      for k = 1, self.beamSize do
        if completed[b][k] == 1 then
          local hypothesis = self:_getHypothesis(beams, b, k)
          beams[t]:addCompletedHypotheses(hypothesis, origId)
        end
      end
    else
      table.insert(finishedBatches, origId)
      for k = 1, nBest do
        local hypothesis = self:_getHypothesis(beams, b, k)
        table.insert(hypotheses, hypothesis)
      end
    end
  end

  -- Remove finished batches
  if remainingId < batchSize then
    beams[t]:removeFinishedBatches(remainingIds)
  end
  return finishedBatches, hypotheses
end

return BeamSearcher
