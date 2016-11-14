require 'torch'

local function adagrad_step(x, dfdx, lr, state)
  if not state.var then
    state.var = torch.Tensor():typeAs(x):resizeAs(x):zero()
    state.std = torch.Tensor():typeAs(x):resizeAs(x)
  end

  state.var:addcmul(1, dfdx, dfdx)
  state.std:sqrt(state.var)
  x:addcdiv(-lr, dfdx, state.std:add(1e-10))
end

local function adam_step(x, dfdx, lr, state)
  local beta1 = state.beta1 or 0.9
  local beta2 = state.beta2 or 0.999
  local eps = state.eps or 1e-8

  state.t = state.t or 0
  state.m = state.m or x.new(dfdx:size()):zero()
  state.v = state.v or x.new(dfdx:size()):zero()
  state.denom = state.denom or x.new(dfdx:size()):zero()

  state.t = state.t + 1
  state.m:mul(beta1):add(1-beta1, dfdx)
  state.v:mul(beta2):addcmul(1-beta2, dfdx, dfdx)
  state.denom:copy(state.v):sqrt():add(eps)

  local bias1 = 1-beta1^state.t
  local bias2 = 1-beta2^state.t
  local stepSize = lr * math.sqrt(bias2)/bias1
  x:addcdiv(-stepSize, state.m, state.denom)

end

local function adadelta_step(x, dfdx, lr, state)
  local rho = state.rho or 0.9
  local eps = state.eps or 1e-6
  state.var = state.var or x.new(dfdx:size()):zero()
  state.std = state.std or x.new(dfdx:size()):zero()
  state.delta = state.delta or x.new(dfdx:size()):zero()
  state.accDelta = state.accDelta or x.new(dfdx:size()):zero()
  state.var:mul(rho):addcmul(1-rho, dfdx, dfdx)
  state.std:copy(state.var):add(eps):sqrt()
  state.delta:copy(state.accDelta):add(eps):sqrt():cdiv(state.std):cmul(dfdx)
  x:add(-lr, state.delta)
  state.accDelta:mul(rho):addcmul(1-rho, state.delta, state.delta)
end


local Optim = torch.class("Optim")

function Optim:__init(args)
  self.val_perf = {}

  self.method = args.method
  self.learning_rate = args.learning_rate

  if self.method == 'sgd' then
    self.lr_decay = args.lr_decay
    self.start_decay = false
    self.start_decay_at = args.start_decay_at
  else
    if args.optim_states ~= nil then
      self.optim_states = args.optim_states
    else
      self.optim_states = {}
      for j = 1, args.num_models do
        self.optim_states[j] = {}
      end
    end
  end
end

function Optim:update_params(params, grad_params, max_grad_norm)
  -- compute gradients norm
  local grad_norm = 0
  for j = 1, #grad_params do
    grad_norm = grad_norm + grad_params[j]:norm()^2
  end
  grad_norm = math.sqrt(grad_norm)

  local shrinkage = max_grad_norm / grad_norm

  for j = 1, #grad_params do
    -- normalize gradients
    if shrinkage < 1 then
      grad_params[j]:mul(shrinkage)
    end

    -- update params according to the optimization method
    if self.method == 'adagrad' then
      adagrad_step(params[j], grad_params[j], self.learning_rate, self.optim_states[j])
    elseif self.method == 'adadelta' then
      adadelta_step(params[j], grad_params[j], self.learning_rate, self.optim_states[j])
    elseif self.method == 'adam' then
      adam_step(params[j], grad_params[j], self.learning_rate, self.optim_states[j])
    else
      params[j]:add(grad_params[j]:mul(-self.learning_rate))
    end

    -- zero gradients
    grad_params[j]:zero()
  end
end

-- decay learning rate if val perf does not improve or we hit the start_decay_at limit
function Optim:update_learning_rate(score, epoch)
  self.val_perf[#self.val_perf + 1] = score

  if epoch >= self.start_decay_at then
    self.start_decay = true
  end

  if self.val_perf[#self.val_perf] ~= nil and self.val_perf[#self.val_perf-1] ~= nil then
    local curr_ppl = self.val_perf[#self.val_perf]
    local prev_ppl = self.val_perf[#self.val_perf-1]
    if curr_ppl > prev_ppl then
      self.start_decay = true
    end
  end

  if self.start_decay then
    self.learning_rate = self.learning_rate * self.lr_decay
  end
end

function Optim:get_learning_rate()
  return self.learning_rate
end

function Optim:get_states()
  return self.optim_states
end

return Optim
