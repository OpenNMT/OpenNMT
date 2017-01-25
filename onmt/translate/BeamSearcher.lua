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
  * `beforeFilterFactor` - optional, set this only if filter is being used. Before applying filters, hypotheses with top `beamSize * preFilterFactor` scores will be considered. If the returned hypotheses voilate filters, then set this to a larger value to consider more. [1]
  * `keepInitial` - optional, whether return the initial token or not. [false]

Returns: a table `finished`. `finished[b][n].score`, `finished[b][n].tokens` and `finished[b][n].states` describe the n-th best hypothesis for b-th sample in the batch.

]]
function BeamSearcher:search(beamSize, nBest, beforeFilterFactor, keepInitial)
  self.nBest = nBest or 1
  self.beamSize = beamSize or 1
  self.beforeFilterFactor = beforeFilterFactor or 1
  self.keepInitial = keepInitial or false

  local beams = {}
  local finished = {}

  -- Initialize the beam.
  beams[1] = self.advancer:initBeam()
  local remaining = beams[1]:remaining()
  if beams[1]:tokens()[1]:size(1) ~= remaining * beamSize then
    beams[1]:replicate(self.beamSize)
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
    local completed = self.advancer:isComplete(beams[t + 1])

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

function BeamSearcher:_retrieveHypothesis(beams, batchId, score, tok, bp, t)
  local states = {}
  local tokens = {}

  tokens[t - 1] = tok
  t = t - 1
  local remainingId
  while t > 0 do
    if t == 1 then
      remainingId = batchId
    else
      remainingId = beams[t]:orig2Remaining(batchId)
    end
    states[t] = beams[t]:indexState(self.beamSize, remainingId, bp,
                                    self.advancer.keptStateIndexes)
    tokens[t - 1] = beams[t]:indexToken(self.beamSize, remainingId, bp)
    bp = beams[t]:indexBackPointer(self.beamSize, remainingId, bp)
    t = t - 1
  end
  if not self.keepInitial then
    tokens[0] = nil
  end

  -- Transpose states
  local statesTemp = {}
    for r = 1, #states do
      for j, _ in pairs(states[r]) do
        statesTemp[j] = statesTemp[j] or {}
        statesTemp[j][r] = states[r][j]
      end
    end
  states = statesTemp
  return {tokens = tokens, states = states, score = score}
end

-- Find the top beamSize hypotheses (satisfying filters)
function BeamSearcher:_findKBest(beams, scores)
  local t = #beams
  local vocabSize = scores:size(2)
  local expandedScores = beams[t]:expandScores(scores, self.beamSize)

  -- Find top beamSize * beforeFilterFactor hypotheses
  local considered = self.beamSize * self.beforeFilterFactor
  local consideredScores, consideredIds = expandedScores:topk(considered, 2,
                                                              true, true)
  consideredIds:add(-1)
  local consideredBackPointer = (consideredIds:clone():div(vocabSize)):add(1)
  local consideredToken = consideredIds:fmod(vocabSize):add(1):view(-1)

  local newBeam = beams[t]:nextBeam(consideredToken, consideredScores,
                                    consideredBackPointer, self.beamSize)

  -- Prune hypotheses if necessary
  local pruned = self.advancer:filter(newBeam)
  if pruned and pruned:any() then
    consideredScores:view(-1):maskedFill(pruned, -math.huge)
  end

  -- Find top beamSize hypotheses
  if ( (not pruned) or (not pruned:any()) ) and (self.beforeFilterFactor == 1) then
    beams[t + 1] = newBeam
  else
    local kBestScores, kBestIds = consideredScores:topk(self.beamSize, 2,
                                                        true, true)
    local backPointer = consideredBackPointer:gather(2, kBestIds)
    local token = consideredToken:viewAs(consideredIds)
                                 :gather(2, kBestIds):view(-1)
    newBeam = beams[t]:nextBeam(token, kBestScores,
                                backPointer, self.beamSize)
    beams[t + 1] = newBeam
  end
end

function BeamSearcher:_completeHypotheses(beams, completed)
  local t = #beams
  local batchSize = beams[t]:remaining()
  completed = completed:view(batchSize, -1)
  local token = beams[t]:tokens()[t]:view(batchSize, -1)
  local backPointer = beams[t]:backPointer():view(batchSize, -1)
  local scores = beams[t]:scores():view(batchSize, -1)

  local remainingId = 0
  local remainingIds = {}
  local finishedBatches = {}
  local finishedHypotheses = {}
  for b = 1, batchSize do
    local origId = b
    if t > 2 then
      origId = beams[t - 1]:remaining2Orig(b)
      assert (origId)
    end
    local prevCompleted = onmt.translate.Beam.completed(origId)
    local batchFinished = true
    local prevId = 1
    local currId = 1
    for _ = 1, self.nBest do
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
      else
        if completed[b][currId] == 0 then
          batchFinished = false
          break
        end
        currId = currId + 1
      end
    end
    if not batchFinished then
      remainingId = remainingId + 1
      beams[t]:setOrig2Remaining(origId, remainingId)
      beams[t]:setRemaining2Orig(remainingId, origId)
      table.insert(remainingIds, b)
      for k = 1, self.beamSize do
        if completed[b][k] == 1 then
          local tok = token[b][k]
          local bp = backPointer[b][k]
          local score = scores[b][k]
          onmt.translate.Beam.addCompletedHypotheses(score, tok, bp, t, origId)
        end
      end
    else
      table.insert(finishedBatches, origId)
      local hypothesis = {}
      prevId = 1
      currId = 1
      for _ = 1, self.nBest do
        if prevId <= #prevCompleted then
          local prevScore = prevCompleted[prevId][1]
          if prevScore > scores[b][currId] then
            table.insert(hypothesis,
                         self:_retrieveHypothesis(beams, origId,
                                   table.unpack(prevCompleted[prevId])))
            prevId = prevId + 1
          else
            assert( completed[b][currId] == 1 )
            local score = scores[b][currId]
            local tok = token[b][currId]
            local bp = backPointer[b][currId]
            table.insert(hypothesis,
                         self:_retrieveHypothesis(beams, origId, score, tok, bp, t))
            currId = currId + 1
          end
        else
          assert( completed[b][currId] == 1 )
          local score = scores[b][currId]
          local tok = token[b][currId]
          local bp = backPointer[b][currId]
          table.insert(hypothesis,
                       self:_retrieveHypothesis(beams, origId, score, tok, bp, t))
          currId = currId + 1
        end
      end
      table.insert(finishedHypotheses, hypothesis)
      onmt.translate.Beam.removeCompleted(origId)
    end
  end

  beams[t]:scores():maskedFill(completed:view(-1), -math.huge)

  -- Remove finished batches
  if remainingId < batchSize then
    beams[t]:removeFinishedBatches(remainingIds, self.beamSize)
  end
  return finishedBatches, finishedHypotheses
end

return BeamSearcher
