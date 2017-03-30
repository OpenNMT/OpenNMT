---------------------------------------------------------------------------------
-- Local utility functions
---------------------------------------------------------------------------------

local function adagradStep(dfdx, lr, state)
  if not state.var then
    state.var = torch.Tensor():typeAs(dfdx):resizeAs(dfdx):zero()
    state.std = torch.Tensor():typeAs(dfdx):resizeAs(dfdx)
  end

  state.var:addcmul(1, dfdx, dfdx)
  state.std:sqrt(state.var)
  dfdx:cdiv(state.std:add(1e-10)):mul(-lr)
end

local function adamStep(dfdx, lr, state)
  local beta1 = state.beta1 or 0.9
  local beta2 = state.beta2 or 0.999
  local eps = state.eps or 1e-8

  state.t = state.t or 0
  state.m = state.m or dfdx.new(dfdx:size()):zero()
  state.v = state.v or dfdx.new(dfdx:size()):zero()
  state.denom = state.denom or dfdx.new(dfdx:size()):zero()

  state.t = state.t + 1
  state.m:mul(beta1):add(1-beta1, dfdx)
  state.v:mul(beta2):addcmul(1-beta2, dfdx, dfdx)
  state.denom:copy(state.v):sqrt():add(eps)

  local bias1 = 1-beta1^state.t
  local bias2 = 1-beta2^state.t
  local stepSize = lr * math.sqrt(bias2)/bias1

  dfdx:copy(state.m):cdiv(state.denom):mul(-stepSize)
end

local function adadeltaStep(dfdx, state)
  local rho = state.rho or 0.9
  local eps = state.eps or 1e-6
  state.var = state.var or dfdx.new(dfdx:size()):zero()
  state.std = state.std or dfdx.new(dfdx:size()):zero()
  state.delta = state.delta or dfdx.new(dfdx:size()):zero()
  state.accDelta = state.accDelta or dfdx.new(dfdx:size()):zero()
  state.var:mul(rho):addcmul(1-rho, dfdx, dfdx)
  state.std:copy(state.var):add(eps):sqrt()
  state.delta:copy(state.accDelta):add(eps):sqrt():cdiv(state.std):cmul(dfdx)
  dfdx:copy(state.delta):mul(-1)
  state.accDelta:mul(rho):addcmul(1-rho, state.delta, state.delta)
end

---------------------------------------------------------------------------------

local Optim = torch.class('Optim')

local options = {
  {'-max_batch_size',     64   , [[Maximum batch size]],
                                 {valid=onmt.utils.ExtendedCmdLine.isUInt()}},
  {'-uneven_batches',     false, [[If true, batches are filled up to max_batch_size even if they have different sizes.]]},
  {'-optim',              'sgd', [[Optimization method.]],
                                 {enum={'sgd', 'adagrad', 'adadelta', 'adam'}}},
  {'-learning_rate',       1   , [[Starting learning rate. If adagrad or adam is used,
                                      then this is the global learning rate. Recommended settings are: sgd = 1,
                                      adagrad = 0.1, adam = 0.0002]]},
  {'-min_learning_rate',   0   , [[Do not continue the training past this learning rate]]},
  {'-max_grad_norm',       5   , [[If the norm of the gradient vector exceeds this renormalize it to have
                                       the norm equal to max_grad_norm]]},
  {'-learning_rate_decay', 0.5 , [[Learning rate decay factor]]},
  {'-start_decay_at',      9   , [[With 'default' decay mode, start decay after this epoch]],
                                 {valid=onmt.utils.ExtendedCmdLine.isUInt()}},
  {'-start_decay_ppl_delta', 0 , [[Start decay when validation perplexity improvement is lower than this value]]},
  {'-decay',          'default', [[When to apply learning rate decay.
                                 'default': decay after each epoch past start_decay_at or as soon as the validation perplexity is not improving more than start_decay_ppl_delta,
                                 'perplexity_only': only decay when validation perplexity is not improving more than start_decay_ppl_delta]],
                                 {enum={'default', 'perplexity_only'}}}
}

function Optim.declareOpts(cmd)
  cmd:setCmdLineOptions(options, 'Optimization')
end

function Optim:__init(args, optimStates)
  self.args = onmt.utils.ExtendedCmdLine.getModuleOpts(args, options)
  self.valPerf = {}

  if self.args.optim == 'sgd' then
    self.args.start_decay_at = args.start_decay_at
  else
    if optimStates ~= nil then
      self.optimStates = optimStates
    else
      self.optimStates = {}
    end
  end
end

function Optim:setOptimStates(num)
  if self.args.optim ~= 'sgd' then
    for j = 1, num do
      self.optimStates[j] = {}
    end
  end
end

function Optim:zeroGrad(gradParams)
  for j = 1, #gradParams do
    gradParams[j]:zero()
  end
end

function Optim:prepareGrad(gradParams)
  -- Compute gradients norm.
  local gradNorm = 0
  for j = 1, #gradParams do
    gradNorm = gradNorm + gradParams[j]:norm()^2
  end
  gradNorm = math.sqrt(gradNorm)

  local shrinkage = self.args.max_grad_norm / gradNorm

  for j = 1, #gradParams do
    -- Shrink gradients if needed.
    if shrinkage < 1 then
      gradParams[j]:mul(shrinkage)
    end

    -- Prepare gradients params according to the optimization method.
    if self.args.optim == 'adagrad' then
      adagradStep(gradParams[j], self.args.learning_rate, self.optimStates[j])
    elseif self.args.optim == 'adadelta' then
      adadeltaStep(gradParams[j], self.optimStates[j])
    elseif self.args.optim == 'adam' then
      adamStep(gradParams[j], self.args.learning_rate, self.optimStates[j])
    else
      gradParams[j]:mul(-self.args.learning_rate)
    end
  end
end

function Optim:updateParams(params, gradParams)
  for j = 1, #params do
    params[j]:add(gradParams[j])
  end
end

-- decay learning rate if val perf does not improve or we hit the startDecayAt limit
function Optim:updateLearningRate(score, epoch)
  local function decayLr()
    self.args.learning_rate = self.args.learning_rate * self.args.learning_rate_decay
  end

  if self.args.optim == 'sgd' then
    self.valPerf[#self.valPerf + 1] = score

    if epoch >= self.args.start_decay_at then
      self.startDecay = true
    end

    local decayConditionMet = false

    if self.valPerf[#self.valPerf] ~= nil and self.valPerf[#self.valPerf-1] ~= nil then
      local currPpl = self.valPerf[#self.valPerf]
      local prevPpl = self.valPerf[#self.valPerf-1]
      if prevPpl - currPpl < self.args.start_decay_ppl_delta then
        self.startDecay = true
        decayConditionMet = true
      end
    end

    if self.args.decay == 'default' and self.startDecay then
      decayLr()
    elseif self.args.decay == 'perplexity_only' and decayConditionMet then
      decayLr()
    end

    return self.args.learning_rate >= self.args.min_learning_rate
  end

  return true
end

function Optim:getLearningRate()
  return self.args.learning_rate
end

function Optim:getStates()
  return self.optimStates
end

return Optim
