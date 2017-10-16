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

-- Find the top beamSize hypotheses (satisfying filters).
function BeamSearcher:_findKBest(beams, scores)

  local function topk(tensor, ...)
    if torch.typename(tensor) == 'torch.CudaHalfTensor' then
      tensor = tensor:cuda()
    end
    return tensor:topk(...)
  end

  local t = #beams
  local vocabSize = scores:size(2)
  local expandedScores, expandedNormScores = beams[t]:_expandScores(scores, self.beamSize)

  -- Find top beamSize * preFilterFactor hypotheses.
  local considered = self.beamSize * self.preFilterFactor
  local consideredNormScores, consideredIds = topk(expandedNormScores, considered, 2, true, true)
  local consideredScores = expandedScores:gather(2, consideredIds)

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

  local newBeam = beams[t]:_nextBeam(consideredToken, consideredScores,
                                    consideredBackPointer, self.beamSize)

  -- Prune hypotheses if necessary.
  local pruned = self.advancer:filter(newBeam)
  if pruned and pruned:any() then
    consideredScores:view(-1):maskedFill(pruned, -math.huge)
    consideredNormScores:view(-1):maskedFill(pruned, -math.huge)
  end

  -- Find top beamSize hypotheses.
  if ((not pruned) or (not pruned:any())) and (self.preFilterFactor == 1) then
    -- TODO
  else
    local _, kBestIds = topk(consideredNormScores, self.beamSize, 2, true, true)
    consideredScores = consideredScores:gather(2, kBestIds)
    consideredBackPointer = consideredBackPointer:gather(2, kBestIds)
    consideredToken = consideredToken
      :viewAs(consideredIds)
      :gather(2, kBestIds)
      :view(-1)
  end

  local constraints = beams[t]:getState()[11]
  local constraintNum = constraints:size(2)

  -- Constraints not yet used. 0 = constraint is available.
  local availableConstraints = torch.eq(constraints, 1)

  -- Add vocabSize * beam number to constraint indexes
  local expandedConstraints = constraints:view(-1, self.beamSize*constraintNum):clone()
  local vocabBeamIdx = torch.range(0,self.beamSize-1):typeAs(constraints):view(self.beamSize,1):mul(vocabSize):expand(self.beamSize, constraintNum):clone():view(1,-1):expandAs(expandedConstraints)
  expandedConstraints:add(vocabBeamIdx)

  -- Gather scores for available constraints
  local constraintScores = torch.gather(expandedScores, 2, expandedConstraints)

  -- Mask scores for constraints that are not available and choose top scored constraints
  constraintScores:maskedFill(availableConstraints:view(-1, self.beamSize*constraintNum), -math.huge)
  constraintScores, constraintScoreIdx = topk(constraintScores, self.beamSize/2, 2, true, true)

  -- Reverse top constraint scores (since they need to be added at the end of each beam).
  constraintScores = constraintScores:index(2 ,torch.linspace(self.beamSize/2,1,self.beamSize/2):long())
  constraintScoreIdx = constraintScoreIdx:index(2 ,torch.linspace(self.beamSize/2,1,self.beamSize/2):long())

  -- Gather corresponding constraint indexes and beam back pointers
  local topConstraints = constraints:view(-1, self.beamSize*constraintNum):gather(2, constraintScoreIdx)
  local topContraintBP = constraintScoreIdx:clone():add(-1):div(constraintNum):add(1)

  local constraintScoresMask = torch.eq(constraintScores, -math.huge)
  local constraintScoresMaskInv = torch.ne(constraintScores, -math.huge)

  -- Select constraint scores, indexes and back pointers
  constraintScores = constraintScores:maskedSelect(constraintScoresMaskInv)
  topConstraints = topConstraints:maskedSelect(constraintScoresMaskInv)
  topContraintBP = topContraintBP:maskedSelect(constraintScoresMaskInv)

  -- Insert constraints into selected scores, selected tokens and selected backpointers
  consideredScores[{{},{self.beamSize/2+1, self.beamSize}}]:maskedCopy(constraintScoresMaskInv, constraintScores)
  consideredToken:view(-1, self.beamSize)[{{},{self.beamSize/2+1, self.beamSize}}]:maskedCopy(constraintScoresMaskInv, topConstraints)
  consideredBackPointer[{{},{self.beamSize/2+1, self.beamSize}}]:maskedCopy(constraintScoresMaskInv, topContraintBP)

  -- Create a mask for the contraint used at current step
  local constraintIdx = constraintScoreIdx:clone():add(-1):fmod(constraintNum):add(1):view(-1, self.beamSize/2, 1)
  local constraintMask = constraints:clone():zero():typeAs(constraintScoresMask)
  local constraintMaskSlice = constraintMask:view(-1, self.beamSize, constraintNum)[{{},{self.beamSize/2+1, self.beamSize}, {}}]
  constraintMaskSlice:scatter(3,constraintIdx, 1)
  constraintScoresMaskExpanded = constraintScoresMask:view(-1, self.beamSize/2, 1):expandAs(constraintMaskSlice)
  constraintMaskSlice:maskedFill(constraintScoresMaskExpanded,0)

  -- TODO : make it optional
  newBeam = beams[t]:_nextBeam(consideredToken, consideredScores, consideredBackPointer, self.beamSize, constraintMask)
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
