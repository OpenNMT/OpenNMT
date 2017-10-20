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
  * `saveHistory` - if true, save the beam search history.

]]
function BeamSearcher:__init(advancer, saveHistory)
  self.advancer = advancer
  self.saveHistory = saveHistory
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

Returns:

  * a table `finished`. `finished[b][n].score`, `finished[b][n].tokens`
    and `finished[b][n].states` describe the n-th best hypothesis for b-th sample
    in the batch.
  * (optional) a table `histories`. `histories[b].predictedIds`, `histories[b].parentBeams`
    and `histories[b].scores` save the full beam search history of the b-th sample.

]]
function BeamSearcher:search(beamSize, nBest, preFilterFactor, keepInitial)
  self.nBest = nBest or 1
  self.beamSize = beamSize or 1
  assert(self.nBest <= self.beamSize, 'beam size must be greater or equal to the n-best list size')
  self.preFilterFactor = preFilterFactor or 1
  self.keepInitial = keepInitial or false

  local beams = {}
  local finished = {}
  local histories = {}

  -- Initialize the beam.
  beams[1] = self.advancer:initBeam()

  -- If we use lexical constraints, we need as many beams as the number of used constraints
  if beams[1]:getState()[11] then
    self.beamSize = self.beamSize * (beams[1]:getState()[11]:size(2)+1)
  end


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

    -- Find next best tokens and create a new beam (maintained by BeamSearcher).
    self:_makeNewBeam(beams, scores)

    -- Determine which hypotheses are complete.
    local completed = self.advancer:isComplete(beams[t + 1])

    -- Remove completed hypotheses (maintained by BeamSearcher).
    local finishedBatches, finishedHypotheses, finishedHistory =
      self:_completeHypotheses(beams, completed)

    for b = 1, #finishedBatches do
      finished[finishedBatches[b]] = finishedHypotheses[b]
      histories[finishedBatches[b]] = finishedHistory[b]
    end
    t = t + 1
    remaining = beams[t]:getRemaining()
  end
  return finished, histories
end

-- Generate kBest tokens based on top scores.
function BeamSearcher:_findKBest(beams, vocabSize, kBest, expandedScores, expandedNormScores, expandedConstraints)

  local function topk(tensor, ...)
    if torch.typename(tensor) == 'torch.CudaHalfTensor' then
      tensor = tensor:cuda()
    end
    return tensor:topk(...)
  end

  local t = #beams

  -- Find top kBest * preFilterFactor hypotheses.
  local considered = kBest * self.preFilterFactor
  local consideredNormScores, consideredIds = topk(expandedNormScores, considered, 2, true, true)
  local consideredScores = expandedScores:gather(2, consideredIds)

  local batchSize = expandedScores:size(1)

  -- Find corresponding used lexical constraints
  local consideredConstraints
  if expandedConstraints then
    local expandedConsideredIds = consideredIds:view(batchSize, considered, 1):expand(batchSize, considered, expandedConstraints:size(4))
    consideredConstraints = expandedConstraints:view(batchSize, expandedScores:size(2), -1):gather(2, expandedConsideredIds)
  end

  consideredIds:add(-1)

  local consideredBackPointer = (consideredIds:clone():div(vocabSize)):add(1)
  local consideredToken = consideredIds:view(-1)
  if consideredToken.fmod then
    consideredToken = consideredToken:fmod(vocabSize):add(1)
  else
    for i = 1, consideredToken:size(1) do
      consideredToken[i] = math.fmod(consideredToken[i], vocabSize) + 1
    end
  end

  -- substitute with complete dictionary index
  if self.advancer.dicts.subdict then
    self.advancer.dicts.subdict:fullIdx(consideredToken)
  end

  -- TODO : do the filtering without creating a new beam ?
  local newBeam = beams[t]:_nextBeam(consideredToken, consideredScores,
                                    consideredBackPointer, self.beamSize)

  -- Prune hypotheses if necessary.
  local pruned = self.advancer:filter(newBeam)
  if pruned and pruned:any() then
    consideredScores:view(-1):maskedFill(pruned, -math.huge)
    consideredNormScores:view(-1):maskedFill(pruned, -math.huge)
  end

  -- Find top kBest hypotheses.
  if (pruned and pruned:any()) or self.preFilterFactor ~= 1 then
    local _, kBestIds = topk(consideredNormScores, kBest, 2, true, true)
    consideredScores = consideredScores:gather(2, kBestIds)
    consideredBackPointer = consideredBackPointer:gather(2, kBestIds)

    if expandedConstraints then
      local expandedkBestIds = kBestIds:view(batchSize, kBest, 1):expand(batchSize, kBest, expandedConstraints:size(4))
      consideredConstraints = consideredConstraints:gather(2, expandedkBestIds)
    end

    consideredToken = consideredToken
      :viewAs(consideredIds)
      :gather(2, kBestIds)
      :view(-1)
  end

  return consideredScores, consideredBackPointer, consideredToken, consideredConstraints

end

-- Find the top beamSize hypotheses (satisfying filters).
function BeamSearcher:_makeNewBeam(beams, scores)

  local t = #beams
  local vocabSize = scores:size(2)
  local batchSize = beams[t]:getRemaining()

  local expandedScores, expandedNormScores = beams[t]:_expandScores(scores, self.beamSize)

  local newBeamScores = torch.Tensor()
  local newBeamBackPointer = torch.Tensor()
  local newBeamToken = torch.Tensor()
  local newBeamConstraints = nil

  local expandedConstraints, expandedConstraintSizes = beams[t]:_expandConstraints(self.beamSize, vocabSize)

  local constraintNum = 0 
  local usedConstraintNum = nil

  if expandedConstraints and expandedConstraintSizes then
    constraintNum = expandedConstraints:size(4)
    -- number of used constraints for each hypothesis
    usedConstraintNum = expandedConstraintSizes:csub(expandedConstraints:ne(0):sum(4):typeAs(expandedConstraintSizes))
    newBeamConstraints = torch.Tensor()
  end

  -- for each possible number of used constraints, get kbest scores and create a part of the new beam
  for i = 0, constraintNum do
    local maskedScores, maskedNormScores

    if usedConstraintNum then
      -- mask scores for all other number of used constraints
      maskedScores  = expandedScores:clone()
      maskedNormScores = expandedNormScores:clone()

      maskedScores:maskedFill(usedConstraintNum:ne(i), -math.huge)
      maskedNormScores:maskedFill(usedConstraintNum:ne(i), -math.huge)
    else
      maskedScores = expandedScores
      maskedNormScores = expandedNormScores
    end

    -- get kbest scores
    local consideredScores, consideredBackPointer, consideredToken, consideredConstraints = self:_findKBest(beams, vocabSize, self.beamSize/(constraintNum+1), maskedScores, maskedNormScores, expandedConstraints)

    newBeamScores = newBeamScores:typeAs(consideredScores):cat(consideredScores, 2)
    newBeamBackPointer = newBeamBackPointer:typeAs(consideredBackPointer):cat(consideredBackPointer,2)
    newBeamToken = newBeamToken:typeAs(consideredToken):cat(consideredToken:view(batchSize,-1),2)
    if consideredConstraints then
      newBeamConstraints = newBeamConstraints:typeAs(consideredConstraints):cat(consideredConstraints,2)
    end
  end

  if newBeamConstraints then
    newBeamConstraints = newBeamConstraints:view(-1, constraintNum)
  end

  newBeam = beams[t]:_nextBeam(newBeamToken:view(-1), newBeamScores, newBeamBackPointer, self.beamSize, newBeamConstraints)
  beams[t + 1] = newBeam

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

-- Retrieve the complete beam history of batchId.
function BeamSearcher:_retrieveHistory(beams, batchId, t)
  local predictedIds = {}
  local parentBeams = {}
  local scores = {}

  t = t - 1

  while t > 1 do
    local remainingId = beams[t]:orig2Remaining(batchId)

    table.insert(predictedIds, 1, beams[t]:_indexToken(self.beamSize, remainingId):squeeze())
    table.insert(scores, 1, beams[t]:_indexScore(self.beamSize, remainingId):squeeze())
    table.insert(parentBeams, 1, beams[t]:_indexBackPointer(self.beamSize, remainingId):squeeze())

    t = t - 1
  end

  return { predictedIds = predictedIds, parentBeams = parentBeams, scores = scores }
end

-- Checks which sequences are finished and moves finished hypothese to a buffer.
function BeamSearcher:_completeHypotheses(beams, completed)
  local t = #beams
  local batchSize = beams[t]:getRemaining()
  completed = completed:view(batchSize, -1)

  local finishedBatches = {}
  local finishedHypotheses = {}
  local finishedHistory = {}

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

      if self.saveHistory then
        local history = self:_retrieveHistory(beams, origId, t)
        table.insert(finishedHistory, history)
      end

      onmt.translate.Beam._removeCompleted(origId)
    end
  end

  beams[t]:getScores():maskedFill(completed:view(-1), -math.huge)

  -- Remove finished sequences from batch.
  if #remainingIds < batchSize then
    beams[t]:_removeFinishedBatches(remainingIds, self.beamSize)
  end
  return finishedBatches, finishedHypotheses, finishedHistory
end

return BeamSearcher
