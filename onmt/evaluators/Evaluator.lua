--[[ Base class for evaluators. ]]
local Evaluator = torch.class('Evaluator')

--[[ Create a new Evaluator. ]]
function Evaluator:__init()
end

--[[ Run the evaluator on a dataset.

Parameters:

  * `model` - the model to evaluate.
  * `data` - the `Dataset` to evaluate on.
  * `saveFile` - optional filename to save the translation.

Returns: the evaluation score.

]]
function Evaluator:eval(_, _, _)
  error('Not implemented')
end

--[[ Return true if the evaluator can save the translation result. ]]
function Evaluator:canSaveTranslation()
  return false
end

--[[ Compare two scores as returned by the evaluator.

Also see `Evaluator.lowerIsBetter` and `Evaluator.higherIsBetter`.

Parameters:

  * `a` - the score to compare.
  * `b` - the score to compare against.
  * `delta` - the error margin to tolerate.

Returns: `true` if `a` is not worse than `b`, `false` otherwise.

]]
function Evaluator:compare(_, _, _)
  error('Not implemented')
end

-- Predefine common comparison methods.
function Evaluator.lowerIsBetter(a, b, delta)
  delta = delta or 0
  return a - (b - delta) <= 0
end
function Evaluator.higherIsBetter(a, b, delta)
  delta = delta or 0
  return a - (b + delta) >= 0
end

--[[ Return the name of the evaluation metric. ]]
function Evaluator:__tostring__()
  error('Not implemented')
end

return Evaluator
