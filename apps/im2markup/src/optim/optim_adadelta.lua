--[[ ADADELTA implementation for SGD http://arxiv.org/abs/1212.5701

ARGS:
- `opfunc` : a function that takes a single input (X), the point of
            evaluation, and returns f(X) and df/dX
- `x` : the initial point
- `config` : a table of hyper-parameters
- `config.rho` : interpolation parameter
- `config.eps` : for numerical stability
- `config.weightDecay` : weight decay
- `state` : a table describing the state of the optimizer; after each
         call the state is modified
- `state.paramVariance` : vector of temporal variances of parameters
- `state.accDelta` : vector of accummulated delta of gradients
RETURN:
- `x` : the new x vector
- `f(x)` : the function, evaluated before the update
]]
function optim.adadelta_list(opfunc, x, config, state)
    -- (0) get/update state
    if config == nil and state == nil then
        print('no state table, ADADELTA initializing')
    end
    local config = config or {}
    local state = state or config
    local rho = config.rho or 0.9
    local eps = config.eps or 1e-6
    local wd = config.weightDecay or 0
    -- (1) evaluate f(x) and df/dx
    local fx,dfdx,stats = opfunc(x)

    for i = 1, #x do
        -- (2) weight decay
        local y = x[i]
        local dfdy = dfdx[i]
        if wd ~= 0 then
          dfdy:add(wd, y)
        end

        -- (3) parameter update
        if state[i] == nil then
            state[i] = {}
        end
        state[i].evalCounter = state.evalCounter or 0
        if not state[i].paramVariance then
            state[i].paramVariance = torch.Tensor():typeAs(y):resizeAs(dfdy):zero()
            state[i].paramStd = torch.Tensor():typeAs(y):resizeAs(dfdy):zero()
            state[i].delta = torch.Tensor():typeAs(y):resizeAs(dfdy):zero()
            state[i].accDelta = torch.Tensor():typeAs(y):resizeAs(dfdy):zero()
        end
        state[i].paramVariance:mul(rho):addcmul(1-rho,dfdy,dfdy)
        state[i].paramStd:resizeAs(state[i].paramVariance):copy(state[i].paramVariance):add(eps):sqrt()
        state[i].delta:resizeAs(state[i].paramVariance):copy(state[i].accDelta):add(eps):sqrt():cdiv(state[i].paramStd):cmul(dfdy)
        y:add(-0.01, state[i].delta)
        state[i].accDelta:mul(rho):addcmul(1-rho, state[i].delta, state[i].delta)
        -- (4) update evaluation counter
        state[i].evalCounter = state[i].evalCounter + 1
    end
        -- return x*, f(x) before optimization
    return x,{fx}, stats
end

