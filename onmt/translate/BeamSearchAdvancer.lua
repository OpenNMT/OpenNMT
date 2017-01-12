--[[ Class for specifying how to advance one step. A hypothesis consists of a list
   of tokens and a state. In the code, hypotheses are stored as a table of two
   objects, a table representing tokens, and an abstract state object.

  Pseudocode:

      finished = []

      -- Initialize hypotheses

      [hypotheses] <-- init()

      WHILE [hypotheses] is not empty DO

        -- Update hypothesis states based on new tokens

        update([hypotheses])

        -- Expand hypotheses by all possible tokens and return the scores

        [ [scores] ] <-- expand([hypotheses])

        -- Find k best next hypotheses (maintained by BeamSearcher)

        _findKBest([hypotheses], [ [scores] ])

        -- Remove completed hypotheses (maintained by BeamSearcher)

        completed <-- isComplete([hypotheses])

        finished += _completeHypotheses([hypotheses], completed)

      ENDWHILE

 ==================================================================
--]]
local BeamSearchAdvancer = torch.class('BeamSearchAdvancer')

--[[Initialize function. Returns an initial beam.

Returns: `hypotheses`, where .

]]
function BeamSearchAdvancer:initHypotheses()
end

--[[Update function. Update hypothesis states given new tokens

Parameters:

  * `tokens` - a flat tensor of size `batchSize`, the selected extension indexes from the previous step.
  * `states` - a table containing the states from the previous step.

]]
function BeamSearchAdvancer:update(beam)
end

--[[Expand function. Given states, returns the scores of the possible
  `extensionSize` tokens.
        -- Expand hypotheses by all possible tokens and return the scores

Parameters:

  * `hypotheses` - the current states.

Returns:

  * `scores` - a 2D tensor of size `(batchSize, extensionSize)`.

]]
function BeamSearchAdvancer:expand()
end

--[[Determines which hypotheses are complete.

Parameters:

  * `hypotheses` - a table. `hypotheses[t]` is a tensor of size `(batchSize, extensionSize)`, indicating the hypotheses at step `t`.
  * `states` - a table of current states.

Returns: a binary flat tensor of size `(batchSize)`, indicating which hypotheses are complete.

]]
function BeamSearchAdvancer:isComplete()
end

--[[Specifies which states to keep track of. After beam search, those states
  can be retrieved during all steps along with the predictions.

Parameters:

  * `indexes` - a table of integers, specifying the indexes in the `states` to track.

]]
function BeamSearchAdvancer:setKeptStateIndexes(indexes)
  self.keptStateIndexes = indexes
end

--[[Determines which hypotheses shall be pruned.
--
Parameters:

  * `hypotheses` - a table. `hypotheses[t]` is a tensor of size `(batchSize, extensionSize)`, indicating the hypotheses at step `t`.
  * `states` - a table of current states.

Returns: a binary flat tensor of size `(batchSize)`, indicating which hypotheses shall be pruned.

]]
function BeamSearchAdvancer:filter()
end

return BeamSearchAdvancer
