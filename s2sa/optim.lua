require 'torch'

local Optim = torch.class("Optim")

function Optim:__init(learning_rate, lr_decay, start_decay_at)
  self.val_perf = {}
  self.start_decay = 0
  self.learning_rate = learning_rate
  self.lr_decay = lr_decay
  self.start_decay_at = start_decay_at
end

function Optim:update_params(params, grad_params, max_grad_norm)
  -- compute gradients norm
  local grad_norm = 0
  for j = 1, #grad_params do
    grad_norm = grad_norm + grad_params[j]:norm()^2
  end
  grad_norm = math.sqrt(grad_norm)

  -- normalize gradients and update params
  local shrinkage = max_grad_norm / grad_norm
  for j = 1, #grad_params do
    if shrinkage < 1 then
      grad_params[j]:mul(shrinkage)
    end
    params[j]:add(grad_params[j]:mul(-self:get_rate()))
  end
end

-- decay learning rate if val perf does not improve or we hit the start_decay_at limit
function Optim:update_rate(score, epoch)
  self.val_perf[#self.val_perf + 1] = score

  if epoch >= self.start_decay_at then
    self.start_decay = 1
  end

  if self.val_perf[#self.val_perf] ~= nil and self.val_perf[#self.val_perf-1] ~= nil then
    local curr_ppl = self.val_perf[#self.val_perf]
    local prev_ppl = self.val_perf[#self.val_perf-1]
    if curr_ppl > prev_ppl then
      self.start_decay = 1
    end
  end
  if self.start_decay == 1 then
    self.learning_rate = self.learning_rate * self.lr_decay
  end
end

function Optim:get_rate()
  return self.learning_rate
end

return Optim
