local Optim = torch.class('Optim')

local options = {
  {
    '-max_batch_size', 64,
    [[Maximum batch size.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt()
    }
  },
  {
    '-uneven_batches', false,
    [[If set, batches are filled up to `-max_batch_size` even if the source lengths are different.
      Slower but needed for some tasks.]]
  },
  {
    '-optim', 'sgd',
    [[Optimization method.]],
    {
      enum = {'sgd', 'adagrad', 'adadelta', 'adam'},
      train_state = true
    }
  },
  {
    '-learning_rate', 1,
    [[Initial learning rate. If `adagrad` or `adam` is used, then this is the global learning rate.
      Recommended settings are: `sgd` = 1, `adagrad` = 0.1, `adam` = 0.0002.]],
    {
      train_state = true
    }
  },
  {
    '-min_learning_rate', 0,
    [[Do not continue the training past this learning rate value.]],
    {
      train_state = true
    }
  },
  {
    '-max_grad_norm', 5,
    [[Clip the gradients L2-norm to this value. Set to 0 to disable.]],
    {
      train_state = true
    }
  },
  {
    '-learning_rate_decay', 0.7,
    [[Learning rate decay factor: `learning_rate = learning_rate * learning_rate_decay`.]],
    {
      train_state = true
    }
  },
  {
    '-start_decay_at', 9,
    [[In "default" decay mode, start decay after this epoch.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt(),
      train_state = true
    }
  },
  {
    '-start_decay_score_delta', 0,
    [[Start decay when validation score improvement is lower than this value.]],
    {
      train_state = true
    }
  },
  {
    '-decay', 'default',
    [[When to apply learning rate decay.
      `default`: decay after each epoch past `-start_decay_at` or as soon as the
      validation score is not improving more than `-start_decay_score_delta`,
      `epoch_only`: only decay after each epoch past `-start_decay_at`,
      `score_only`: only decay when validation score is not improving more than
      `-start_decay_ppl_delta`.]],
    {
      enum = {'default', 'epoch_only', 'score_only'},
      train_state = true
    }
  },
  {
    '-decay_method', 'default',
    [[If `restart` is set, the optimizer states (if any) will be reset when the
      decay condition is met.]],
    {
      enum = {'default', 'restart'},
      train_state = true
    }
  }
}

function Optim.declareOpts(cmd)
  cmd:setCmdLineOptions(options, 'Optimization')
end

function Optim:__init(args)
  self.args = onmt.utils.ExtendedCmdLine.getModuleOpts(args, options)
  self.valPerf = {}
end

function Optim:setOptimStates(states)
  self.optimStates = states
end

--[[ Sets optimization states to zero. ]]
function Optim:resetOptimStates()
  if self.optimStates == nil then
    return
  end

  for i = 1, #self.optimStates do
    for key, state in pairs(self.optimStates[i]) do
      if torch.isTensor(state) then
        state:zero()
      elseif key == 't' then
        self.optimStates[i].t = 0
      end
    end
  end
end

function Optim:zeroGrad(gradParams)
  for j = 1, #gradParams do
    gradParams[j]:zero()
  end
end

function Optim:prepareGrad(gradParams)
  if self.args.optim ~= 'sgd' and not self.optimStates then
    self.optimStates = {}
    for _ = 1, #gradParams do
      table.insert(self.optimStates, {})
    end
  end

  if self.args.max_grad_norm > 0 then
    Optim.clipGradByNorm(gradParams, self.args.max_grad_norm)
  end

  for j = 1, #gradParams do
    -- Prepare gradients params according to the optimization method.
    if self.args.optim == 'adagrad' then
      Optim.adagradStep(gradParams[j], self.args.learning_rate, self.optimStates[j])
    elseif self.args.optim == 'adadelta' then
      Optim.adadeltaStep(gradParams[j], self.optimStates[j])
    elseif self.args.optim == 'adam' then
      Optim.adamStep(gradParams[j], self.args.learning_rate, self.optimStates[j])
    else
      gradParams[j]:mul(-self.args.learning_rate)
    end
  end
end

function Optim:status()
  local status = 'Optim ' .. self.args.optim:upper()
  if self.args.optim ~= 'adadelta' then
    status = status .. ' LR '.. string.format("%.6f", self.args.learning_rate)
  end
  return status
end

function Optim:updateParams(params, gradParams)
  for j = 1, #params do
    params[j]:add(gradParams[j])
  end
end

--[[ Update the learning rate if conditions are met (see the documentation).

Parameters:

  * `score` - the last validation score.
  * `èpoch` - the current epoch number.
  * `evaluator` - the `Evaluator` used to compute `score`.

Returns: the new learning rate.

]]
function Optim:updateLearningRate(score, epoch, evaluator)
  local function decayLr()
    self.args.learning_rate = self.args.learning_rate * self.args.learning_rate_decay

    if self.args.decay_method == 'restart' then
      self:resetOptimStates()
    end
  end

  evaluator = evaluator or onmt.evaluators.PerplexityEvaluator.new()

  if self.args.optim == 'sgd' or self.args.optim == 'adam' then
    self.valPerf[#self.valPerf + 1] = score

    local epochCondMet = (epoch >= self.args.start_decay_at)
    local scoreCondMet = false

    if self.valPerf[#self.valPerf] ~= nil and self.valPerf[#self.valPerf-1] ~= nil then
      local currScore = self.valPerf[#self.valPerf]
      local prevScore = self.valPerf[#self.valPerf-1]
      scoreCondMet = not evaluator:compare(currScore, prevScore, self.args.start_decay_score_delta)
    end

    if self.args.decay == 'default' and (epochCondMet or scoreCondMet or self.startDecay) then
      self.startDecay = true
      decayLr()
    elseif self.args.decay == 'epoch_only' and epochCondMet then
      decayLr()
    elseif self.args.decay == 'score_only' and scoreCondMet then
      decayLr()
    end
  end

  return self.args.learning_rate
end

function Optim:isFinished()
  if self.args.optim == 'sgd' or self.args.optim == 'adam' then
    return self.args.learning_rate < self.args.min_learning_rate
  else
    return false
  end
end

function Optim:getLearningRate()
  return self.args.learning_rate
end

function Optim:getStates()
  return self.optimStates
end

--[[ Clips gradients to a maximum L2-norm.

Parameters:

  * `gradParams` - a table of Tensor.
  * `maxNorm` - the maximum L2-norm.

]]
function Optim.clipGradByNorm(gradParams, maxNorm)
  local gradNorm = 0
  for j = 1, #gradParams do
    gradNorm = gradNorm + gradParams[j]:norm()^2
  end
  gradNorm = math.sqrt(gradNorm)

  local clipCoef = maxNorm / gradNorm

  if clipCoef < 1 then
    for j = 1, #gradParams do
      gradParams[j]:mul(clipCoef)
    end
  end
end

function Optim.adagradStep(dfdx, lr, state)
  state.var = state.var or dfdx.new(dfdx:size()):zero()

  local std = dfdx.new(dfdx:size())

  state.var:addcmul(1, dfdx, dfdx)
  std:sqrt(state.var)
  dfdx:cdiv(std:add(1e-10)):mul(-lr)
end

function Optim.adamStep(dfdx, lr, state)
  local beta1 = state.beta1 or 0.9
  local beta2 = state.beta2 or 0.999
  local eps = state.eps or 1e-8

  state.t = state.t or 0
  state.m = state.m or dfdx.new(dfdx:size()):zero()
  state.v = state.v or dfdx.new(dfdx:size()):zero()

  local denom = dfdx.new(dfdx:size())

  state.t = state.t + 1
  state.m:mul(beta1):add(1-beta1, dfdx)
  state.v:mul(beta2):addcmul(1-beta2, dfdx, dfdx)
  denom:copy(state.v):sqrt():add(eps)

  local bias1 = 1-beta1^state.t
  local bias2 = 1-beta2^state.t
  local stepSize = lr * math.sqrt(bias2)/bias1

  dfdx:copy(state.m):cdiv(denom):mul(-stepSize)
end

function Optim.adadeltaStep(dfdx, state)
  local rho = state.rho or 0.9
  local eps = state.eps or 1e-6
  state.var = state.var or dfdx.new(dfdx:size()):zero()
  state.accDelta = state.accDelta or dfdx.new(dfdx:size()):zero()

  local std = dfdx.new(dfdx:size())
  local delta = dfdx.new(dfdx:size())

  state.var:mul(rho):addcmul(1-rho, dfdx, dfdx)
  std:copy(state.var):add(eps):sqrt()
  delta:copy(state.accDelta):add(eps):sqrt():cdiv(std):cmul(dfdx)
  dfdx:copy(delta):mul(-1)
  state.accDelta:mul(rho):addcmul(1-rho, delta, delta)
end

return Optim
