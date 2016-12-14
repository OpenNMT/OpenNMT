--[[ A plain implementation of SGD
ARGS:
- `opfunc` : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX
- `x`      : the initial point
- `config` : a table with configuration parameters for the optimizer
- `config.learningRate`      : learning rate
- `config.learningRateDecay` : learning rate decay
- `config.weightDecay`       : weight decay
- `config.weightDecays`      : vector of individual weight decays
- `config.momentum`          : momentum
- `config.dampening`         : dampening for momentum
- `config.nesterov`          : enables Nesterov momentum
- `config.learningRates`     : vector of individual learning rates
- `state`  : a table describing the state of the optimizer; after each
             call the state is modified
- `state.evalCounter`        : evaluation counter (optional: 0, by default)
RETURN:
- `x`     : the new x vector
- `f(x)`  : the function, evaluated before the update
(Clement Farabet, 2012)
]]
function optim.sgd_list(opfunc, x, config, state)
   -- (0) get/update state
   local config = config or {}
   local state = state or config
   local lr = config.learningRate or 1e-3
   local lrd = config.learningRateDecay or 0
   local wd = config.weightDecay or 0
   local mom = config.momentum or 0
   local damp = config.dampening or mom
   local nesterov = config.nesterov or false
   local lrs = config.learningRates
   local wds = config.weightDecays
   assert(not nesterov or (mom > 0 and damp == 0), "Nesterov momentum requires a momentum and zero dampening")

   -- (1) evaluate f(x) and df/dx
   local fx,dfdx,stats = opfunc(x)

   for i = 1, #x do
       if state[i] == nil then
           state[i] = {}
       end

       state[i].evalCounter = state[i].evalCounter or 0
       local nevals = state[i].evalCounter
       local y = x[i]
       local dfdy = dfdx[i]
       if dfdy:norm() > 5 then
           dfdy:mul(5.0/dfdy:norm())
       end

       -- (2) weight decay with single or individual parameters
       if wd ~= 0 then
          dfdy:add(wd, y)
       elseif wds then
          if not state[i].decayParameters then
             state[i].decayParameters = torch.Tensor():typeAs(x):resizeAs(dfdy)
          end
          state[i].decayParameters:copy(wds):cmul(y)
          dfdy:add(state[i].decayParameters)
       end

       -- (3) apply momentum
       if mom ~= 0 then
          if not state[i].dfdy then
             state[i].dfdy = torch.Tensor():typeAs(dfdy):resizeAs(dfdy):copy(dfdy)
          else
             state[i].dfdy:mul(mom):add(1-damp, dfdy)
          end
          if nesterov then
             dfdy:add(mom, state[i].dfdy)
          else
             dfdy = state[i].dfdy
          end
       end

       -- (4) learning rate decay (annealing)
       local clr = lr / (1 + nevals*lrd)

       -- (5) parameter update with single or individual learning rates
       if lrs then
          if not state[i].deltaParameters then
             state[i].deltaParameters = torch.Tensor():typeAs(y):resizeAs(dfdy)
          end
          state[i].deltaParameters:copy(lrs):cmul(dfdy)
          y:add(-clr, state[i].deltaParameters)
       else
          y:add(-clr, dfdy)
       end

       -- (6) update evaluation counter
       state[i].evalCounter = state[i].evalCounter + 1
   end

       -- return x*, f(x) before optimization
   return x,{fx},stats
end
