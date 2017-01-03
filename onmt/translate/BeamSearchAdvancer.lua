--[[ Class for specifying how to advance one step


           ______________
          / (update beam)\
         /                \
      scores, states       \     .................    nil, nil
        /                   \    /                 \     /
    step()                   \ step()               \  step()
       /                      \ /                    \ /
  nil, nil                 tokens, states        tokens, states
    t = 1                      t = 2             t = max seq len
 ==================================================================

Speicifies how to go one step forward.
--]]
local BeamSearchAdvancer = torch.class('BeamSearchAdvancer')

--[[Step function. BeamSearcher will take care of updating beams and
  transforming states, such that this function only need to take a batched input
  and produce a batched output. All tensors (either parameters or returns,
  including tensors inside tables) must have the same first dimension `batchSize`.
  Note that since BeamSearcher will remove completed hypotheses, the `batchSize`
  does not remain between steps, hence this function cannot assume a static one.

Parameters:

  * `tokens` - a flat tensor of size `batchSize`, predictions from the previous step. For the first step, it would be nil.
  * `states` - the states from the previous step. For the first step, it would be nil.

Returns:

  * `scores` - a 2D tensor of size `(batchSize, vocabSize)`, scores for the current step. For the last step, return nil.
  * `states` - the states for the current step. For the last step, return nil.

]]
function BeamSearchAdvancer:step(tokens, states)
  local scores, statesOut, t
  if tokens == nil then
    -- the first step, batchSize 3, vocabSize 5, hiddenSize 7 and 8
    t = 1
    scores = torch.zeros(3, 5)
    statesOut = {torch.zeros(3, 7), torch.zeros(3, 8), 1}
  else
    -- the following steps, must obtain batchSize from tokens
    local batchSize = tokens:size(1)
    t = states[3]
    scores = torch.zeros(batchSize, 5)
    statesOut = {torch.zeros(batchSize, 7), torch.zeros(batchSize, 8), t + 1}
  end
  if t <= 10 then
    return scores, statesOut
  else
    return nil, nil
  end
end

--[[Specifies which states to keep track of. After beam search, those states
  can be retrieved during all steps along with the predictions.

Parameters:

  * `indexes` - a table of integers, specifying the indexes in the `states` to track.

]]
function BeamSearchAdvancer:setKeptStateIndexes(indexes)
  self.keptStateIndexes = indexes
end

return BeamSearchAdvancer
