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

local function adadeltaStep(dfdx, lr, state)
  local rho = state.rho or 0.9
  local eps = state.eps or 1e-6
  state.var = state.var or dfdx.new(dfdx:size()):zero()
  state.std = state.std or dfdx.new(dfdx:size()):zero()
  state.delta = state.delta or dfdx.new(dfdx:size()):zero()
  state.accDelta = state.accDelta or dfdx.new(dfdx:size()):zero()
  state.var:mul(rho):addcmul(1-rho, dfdx, dfdx)
  state.std:copy(state.var):add(eps):sqrt()
  state.delta:copy(state.accDelta):add(eps):sqrt():cdiv(state.std):cmul(dfdx)
  dfdx:copy(state.delta):mul(-lr)
  state.accDelta:mul(rho):addcmul(1-rho, state.delta, state.delta)
end


local Optim = torch.class("Optim")

function Optim:__init(args)
  self.valPerf = {}

  self.method = args.method
  self.learningRate = args.learningRate
  self.max_grad_norm = args.max_grad_norm

  if self.method == 'sgd' then
    self.learningRateDecay = args.learningRateDecay
    self.startDecay = false
    self.startDecayAt = args.startDecayAt
  else
    if args.optimStates ~= nil then
      self.optimStates = args.optimStates
    else
      self.optimStates = {}
    end
  end
end

function Optim:setOptimStates(num)
  if self.method ~= 'sgd' then
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

  local shrinkage = self.max_grad_norm / gradNorm

  for j = 1, #gradParams do
    -- Shrink gradients if needed.
    if shrinkage < 1 then
      gradParams[j]:mul(shrinkage)
    end

    -- Prepare gradients params according to the optimization method.
    if self.method == 'adagrad' then
      adagradStep(gradParams[j], self.learningRate, self.optimStates[j])
    elseif self.method == 'adadelta' then
      adadeltaStep(gradParams[j], self.learningRate, self.optimStates[j])
    elseif self.method == 'adam' then
      adamStep(gradParams[j], self.learningRate, self.optimStates[j])
    else
      gradParams[j]:mul(-self.learningRate)
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
  if self.method == 'sgd' then
    self.valPerf[#self.valPerf + 1] = score

    if epoch >= self.startDecayAt then
      self.startDecay = true
    end

    if self.valPerf[#self.valPerf] ~= nil and self.valPerf[#self.valPerf-1] ~= nil then
      local currPpl = self.valPerf[#self.valPerf]
      local prevPpl = self.valPerf[#self.valPerf-1]
      if currPpl > prevPpl then
        self.startDecay = true
      end
    end

    if self.startDecay then
      self.learningRate = self.learningRate * self.learningRateDecay
    end
  end
end

function Optim:getLearningRate()
  return self.learningRate
end

function Optim:getStates()
  return self.optimStates
end

function Optim.declareOpts(cmd)
  cmd:option('-max_batch_size', 64, [[Maximum batch size]])
  cmd:option('-optim', 'sgd', [[Optimization method. Possible options are: sgd, adagrad, adadelta, adam]])
  cmd:option('-learning_rate', 1, [[Starting learning rate. If adagrad/adadelta/adam is used,
                                  then this is the global learning rate. Recommended settings are: sgd = 1,
                                  adagrad = 0.1, adadelta = 1, adam = 0.0002]])
  cmd:option('-max_grad_norm', 5, [[If the norm of the gradient vector exceeds this renormalize it to have the norm equal to max_grad_norm]])
  cmd:option('-dropout', 0.3, [[Dropout probability. Dropout is applied between vertical LSTM stacks.]])
  cmd:option('-learning_rate_decay', 0.5, [[Decay learning rate by this much if (i) perplexity does not decrease
                                          on the validation set or (ii) epoch has gone past the start_decay_at_limit]])
  cmd:option('-start_decay_at', 9, [[Start decay after this epoch]])
end

return Optim
