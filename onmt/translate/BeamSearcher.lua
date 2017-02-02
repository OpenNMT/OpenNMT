--[[ Class for managing the internals of the beam search process.


      hyp1---hyp1---hyp1 -hyp1
          \             /
      hyp2 \-hyp2 /-hyp2--hyp2
                 /      \
      hyp3---hyp3---hyp3 -hyp3
      ========================

Takes care of beams.
--]]
local BeamSearcher = torch.class('BeamSearcher')

--[[Constructor

Parameters:

  * `advancer` - an `onmt.translate.Advancer` object.

]]
function BeamSearcher:__init(advancer)
  self.advancer = advancer
end

--[[ Performs beam search.

Parameters:

  * `beamSize` - beam size. [1]
  * `nBest` - the `nBest` top hypotheses will be returned after beam search. [1]
  * `preFilterFactor` - optional, set this only if filter is being used. Before
  applying filters, hypotheses with top `beamSize * preFilterFactor` scores will
  be considered. If the returned hypotheses voilate filters, then set this to a
  larger value to consider more. [1]
  * `keepInitial` - optional, whether return the initial token or not. [false]

Returns: a table `finished`. `finished[b][n].score`, `finished[b][n].tokens`
and `finished[b][n].states` describe the n-th best hypothesis for b-th sample
in the batch.

]]
function BeamSearcher:search(beamSize, nBest, preFilterFactor, keepInitial)
  self.nBest = nBest or 1
  self.beamSize = beamSize or 1
  assert (self.nBest <= self.beamSize)
  self.preFilterFactor = preFilterFactor or 1
  self.keepInitial = keepInitial or false

  local beams = {}
  local finished = {}

  -- Initialize the beam.
  beams[1] = self.advancer:initBeam()
  local remaining = beams[1]:getRemaining()
  if beams[1]:getTokens()[1]:size(1) ~= remaining * beamSize then
    beams[1]:_replicate(self.beamSize)
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
    remaining = beams[t]:getRemaining()
  end
  return finished
end

-- Find the top beamSize hypotheses (satisfying filters).
function BeamSearcher:_findKBest(beams, scores)
  local t = #beams
  local vocabSize = scores:size(2)
  local expandedScores = beams[t]:_expandScores(scores, self.beamSize)

  -- Find top beamSize * preFilterFactor hypotheses.
  local considered = self.beamSize * self.preFilterFactor
  local consideredScores, consideredIds = expandedScores:topk(considered, 2, true, true)
  consideredIds:add(-1)
  local consideredBackPointer = (consideredIds:clone():div(vocabSize)):add(1)
  local consideredToken = consideredIds:fmod(vocabSize):add(1):view(-1)

  local newBeam = beams[t]:_nextBeam(consideredToken, consideredScores,
                                    consideredBackPointer, self.beamSize)

  -- Prune hypotheses if necessary.
  local pruned = self.advancer:filter(newBeam)
  if pruned and pruned:any() then
    consideredScores:view(-1):maskedFill(pruned, -math.huge)
  end

  -- Find top beamSize hypotheses.
  if ((not pruned) or (not pruned:any())) and (self.preFilterFactor == 1) then
    beams[t + 1] = newBeam
  else
    local kBestScores, kBestIds = consideredScores:topk(self.beamSize, 2, true, true)
    local backPointer = consideredBackPointer:gather(2, kBestIds)
    local token = consideredToken
      :viewAs(consideredIds)
      :gather(2, kBestIds)
      :view(-1)
    newBeam = beams[t]:_nextBeam(token, kBestScores, backPointer, self.beamSize)
    beams[t + 1] = newBeam
  end

  -- Cleanup unused memory.
  beams[t]:_cleanUp(self.advancer.keptStateIndexes)
end

-- Do a backward pass to get the tokens and states throughout the history.
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
    assert (remainingId)
    states[t] = beams[t]:_indexState(self.beamSize, remainingId, bp, self.advancer.keptStateIndexes)
    tokens[t - 1] = beams[t]:_indexToken(self.beamSize, remainingId, bp)
    bp = beams[t]:_indexBackPointer(self.beamSize, remainingId, bp)
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

-- Checks which sequences are finished and moves finished hypothese to a buffer.
function BeamSearcher:_completeHypotheses(beams, completed)
  local t = #beams
  local batchSize = beams[t]:getRemaining()
  completed = completed:view(batchSize, -1)

  local finishedBatches = {}
  local finishedHypotheses = {}

  -- Keep track of unfinished batch ids.
  local remainingIds = {}

  -- For each sequence in the batch, check whether it is finished or not.
  for b = 1, batchSize do
    local batchFinished = true
    local hypotheses = beams[t]:_getTopHypotheses(b, self.nBest, completed)

    -- Checks whether the top nBest hypotheses are all finished.
    for k = 1, self.nBest do
      local hypothesis = hypotheses[k]
      if not hypothesis.finished then
        batchFinished = false
        break
      end
    end

    if not batchFinished then
      -- For incomplete sequences, the complete hypotheses will be removed
      -- from beam and saved to buffer.
      table.insert(remainingIds, b)
      beams[t]:_addCompletedHypotheses(b, completed)
    else
      -- For complete sequences, we do a backward pass to retrieve the state
      -- values and tokens throughout the history.
      local origId = beams[t]:_getOrigId(b)
      table.insert(finishedBatches, origId)
      local hypothesis = {}
      for k = 1, self.nBest do
        table.insert(hypothesis, self:_retrieveHypothesis(beams,
                                                          table.unpack(hypotheses[k].hypothesis)))
      end
      table.insert(finishedHypotheses, hypothesis)
      onmt.translate.Beam._removeCompleted(origId)
    end
  end

  beams[t]:getScores():maskedFill(completed:view(-1), -math.huge)

  -- Remove finished sequences from batch.
  if #remainingIds < batchSize then
    beams[t]:_removeFinishedBatches(remainingIds, self.beamSize)
  end
  return finishedBatches, finishedHypotheses
end

return BeamSearcher
