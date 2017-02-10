--[[ Generic Model class. ]]
local Model = torch.class('Model')

local model_options = {
  {'-model_type', 'seq2seq',  [[Type of the model to train.
                              This option impacts all options choices]],
                     {enum={'lm','seq2seq'}}},
  {'-param_init', 0.1, [[Parameters are initialized over uniform distribution with support (-param_init, param_init)]],
                       {valid=function(v) return v>=0 and v<=1 end}}
}

function Model.declareOpts(cmd)
  cmd:setCmdLineOptions(model_options)
end

function Model:__init(args)
  self.args = onmt.ExtendedCmdLine.getModuleOpts(args, model_options)
  self.args.train_from = args.train_from
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

return Model
