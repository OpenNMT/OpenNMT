--[[ Generic Model class. ]]
local Model = torch.class('onmt.Model')

function Model:__init()
  self.models = {}
end

function Model:evaluate()
  for _,m in pairs(self.models) do
    m:evaluate()
  end
end

function Model:training()
  for _,m in pairs(self.models) do
    m:training()
  end
end

function Model:initParams(opt, verbose)
  local numParams = 0
  local params = {}
  local gradParams = {}

  if verbose then
    _G.logger:info('Initializing parameters...')
  end

  -- Order the model table because we need all replicas to have the same order.
  local orderedIndex = {}
  for key in pairs(self.models) do
    table.insert(orderedIndex, key)
  end
  table.sort(orderedIndex)

  for _, key in ipairs(orderedIndex) do
    local mod = self.models[key]
    local p, gp = mod:getParameters()

    if opt.train_from:len() == 0 then
      p:uniform(-opt.param_init, opt.param_init)

      mod:apply(function (m)
        if m.postParametersInitialization then
          m:postParametersInitialization()
        end
      end)
    end

    numParams = numParams + p:size(1)
    table.insert(params, p)
    table.insert(gradParams, gp)
  end

  if verbose then
    _G.logger:info(" * number of parameters: " .. numParams)
  end

  return params, gradParams
end
