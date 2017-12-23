--[[ Class for specifying how to advance one step. A beam mainly consists of
  a list of `tokens` and a `state`. `tokens[t]` stores a flat tensors of size
  `batchSize * beamSize` representing tokens at step `t`. `state` can be either
  a tensor with first dimension size `batchSize * beamSize`, or an iterable
  object containing several such tensors.

  Pseudocode:

      finished = []

      beams = {}

      -- Initialize the beam.

      [ beams[1] ] <-- initBeam()

      FOR t = 1, ... DO

        -- Update beam states based on new tokens.

        update([ beams[t] ])

        -- Expand beams by all possible tokens and return the scores.

        [ [scores] ] <-- expand([ beams[t] ])

        -- Find k best next beams (maintained by BeamSearcher).

        _findKBest([beams], [ [scores] ])

        completed <-- isComplete([ beams[t + 1] ])

        -- Remove completed hypotheses (maintained by BeamSearcher).

        finished += _completeHypotheses([beams], completed)

        IF all(completed) THEN

          BREAK

        END

      ENDWHILE

 ==================================================================
--]]
local Advancer = torch.class('Advancer')

--[[Returns an initial beam.

Returns:

  * `beam` - an `onmt.translate.Beam` object.

]]
function Advancer:initBeam()
end

--[[Updates beam states given new tokens.

Parameters:

  * `beam` - beam with updated token list.

]]
function Advancer:update(beam) -- luacheck: no unused args
end

--[[Expands beam by all possible tokens and returns the scores.

Parameters:

  * `beam` - an `onmt.translate.Beam` object.

Returns:

  * `scores` - a 2D tensor of size `(batchSize * beamSize, numTokens)`.

]]
function Advancer:expand(beam) -- luacheck: no unused args
end

--[[Checks which hypotheses in the beam are already finished.

Parameters:

  * `beam` - an `onmt.translate.Beam` object.

Returns: a binary flat tensor of size `(batchSize * beamSize)`, indicating
  which hypotheses are finished.

]]
function Advancer:isComplete(beam) -- luacheck: no unused args
end

--[[Specifies which states to keep track of. After beam search, those states
  can be retrieved during all steps along with the tokens. This is used
  for memory efficiency.

Parameters:

  * `indexes` - a table of iterators, specifying the indexes in the `states` to track.

]]
function Advancer:setKeptStateIndexes(indexes)
  self.keptStateIndexes = indexes
end

--[[Checks which hypotheses shall be pruned checking at current scores, tokens, backpointer and existing beam stack

Parameters:

  * `beam` current beam
  * `consideredToken` hypotheses tokens
  * `consideredScores` hypotheses scores
  * `consideredBackPointer` back pointer

Returns: a binary flat tensor  indicating which hypothesis shall be pruned.

]]
function Advancer:filter()
end

return Advancer
