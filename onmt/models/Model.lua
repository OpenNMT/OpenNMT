--[[ Generic Model class. ]]
local Model = torch.class('onmt.Model')

function Model:__init(args)
  self.models = {}
  self.args = {
    param_init = args.param_init,
    train_from = args.train_from
  }
end

function Model.declareOpts(cmd)
  cmd:option('-param_init', 0.1, [[Parameters are initialized over uniform distribution with support (-param_init, param_init)]])
  cmd:option('-train_from', '',  [[If training from a checkpoint then this is the path to the pretrained model.]])
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

function Model:initParams(verbose)
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

    if self.args.train_from:len() == 0 then
      p:uniform(-self.args.param_init, self.args.param_init)

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
