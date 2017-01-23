--[[ Class for specifying how to advance one step. A beam mainly consists of
  a list of tokens and a state. Tokens are stored as flat tensors of size
  batchSize, while state can be either a tensor with first dimension size
  batchSize, or an iterable object containing several such tensors.

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


        completed <-- isComplete([ beams[t] ])

        -- Remove completed hypotheses (maintained by BeamSearcher).

        finished += _completeHypotheses([beams], completed)

        IF all(completed) THEN

          BREAK

        END

      ENDWHILE

 ==================================================================
--]]
local BeamSearchAdvancer = torch.class('BeamSearchAdvancer')

--[[Returns an initial beam.

Returns:

  * `beam` - an `onmt.translate.Beam` object.

]]
function BeamSearchAdvancer:initBeam()
end

--[[Updates beam states given new tokens.

Parameters:

  * `beam` - beam with updated token list.

]]
function BeamSearchAdvancer:update(beam)
end

--[[Expand function. Expands beam by all possible tokens and returns the
  scores.

Parameters:

  * `beam` - an `onmt.translate.Beam` object.

Returns:

  * `scores` - a 2D tensor of size `(batchSize, numTokens)`.

]]
function BeamSearchAdvancer:expand(beam)
end

--[[Determines which beams are complete.

Parameters:

  * `beam` - an `onmt.translate.Beam` object.

Returns: a binary flat tensor of size `(batchSize)`, indicating which beams are complete.

]]
function BeamSearchAdvancer:isComplete(beam)
end

--[[Specifies which states to keep track of. After beam search, those states
  can be retrieved during all steps along with the tokens.

Parameters:

  * `indexes` - a table of integers, specifying the indexes in the `states` to track.

]]
function BeamSearchAdvancer:setKeptStateIndexes(indexes)
  self.keptStateIndexes = indexes
end

--[[Determines which beams shall be pruned.
--
Parameters:

  * `beam` - an `onmt.translate.Beam` object.

Returns: a binary flat tensor of size `(batchSize)`, indicating which beams shall be pruned.

]]
function BeamSearchAdvancer:filter()
end

return BeamSearchAdvancer
