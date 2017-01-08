--[[ Class for specifying how to advance one step

  -- Initialize states
  extensions, states <-- init()
  while number of remaining hypotheses > 0 do
    -- Update states
    states <-- forward(extensions, states)
    -- Extend hypotheses by one step and return the scores
    scores <-- expand(states)
    -- Update beam (maintained by BeamSearcher)
    extensions <-- _updateBeam()
    for b = 1, batchSize do
      if isFinal(hypotheses)[b] then
        -- Remove completed hypotheses (maintained by BeamSearcher)
        _removeHypotheses(b)

 ==================================================================

Speicifies how to go one step forward.
--]]
local BeamSearchAdvancer = torch.class('BeamSearchAdvancer')

--[[Constructor

Parameters:

  * `init` - the initialize function.
  * `forward` - the forward function.
  * `expand` - the expand function.
  * `isComplete` - the function that determines complete hypotheses.
  * `filter` - the filter function. Optional.

]]
function BeamSearchAdvancer:__init(init, forward, expand, isComplete, filter)
  self.init = init
  self.forward = forward
  self.expand = expand
  self.isComplete = isComplete
  self.filter = filter
end
--[[Initialize function. Returns initial inputs for the forward function.

Returns:

  * `extensions` - the initial extensions if necessary.
  * `states` - the initial states.

]]
function BeamSearchAdvancer.init()
end

--[[Forward function. Update states given extensions from the previous step.

Parameters:

  * `extensions` - a flat tensor of size `batchSize`, the selected extension indexes from the previous step.
  * `states` - a table containing the states from the previous step.

Returns:

  * `states` - a table containing the updated states.

]]
function BeamSearchAdvancer.forward()
end

--[[Expand function. Given states, returns the scores of the possible
  `extensionSize` extensions.

Parameters:

  * `states` - the current states.

Returns:

  * `scores` - a 2D tensor of size `(batchSize, extensionSize)`.

]]
function BeamSearchAdvancer.expand()
end

--[[Determines which hypotheses are complete.

Parameters:

  * `hypotheses` - a table. `hypotheses[t]` is a tensor of size `(batchSize, extensionSize)`, indicating the hypotheses at step `t`.
  * `states` - a table of current states.

Returns: a binary flat tensor of size `(batchSize)`, indicating which hypotheses are complete.

]]
function BeamSearchAdvancer.isComplete()
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
function BeamSearchAdvancer.filter()
end

return BeamSearchAdvancer
