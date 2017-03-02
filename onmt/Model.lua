--[[ Generic Model class. ]]
local Model = torch.class('Model')

local options = {
  {'-model_type', 'seq2seq',  [[Type of the model to train.
                              This option impacts all options choices]],
                     {enum={'lm','seq2seq', 'seqtagger'}}},
  {'-param_init', 0.1, [[Parameters are initialized over uniform distribution with support (-param_init, param_init)]],
                       {valid=function(v) return v >= 0 and v <= 1 end}}
}

function Model.declareOpts(cmd)
  cmd:setCmdLineOptions(options)
end

function Model:__init(args)
  self.args = onmt.utils.ExtendedCmdLine.getModuleOpts(args, options)
  self.models = {}
end

function Model:getInputLabelsCount(batch)
  return batch.sourceInput:ne(onmt.Constants.PAD):sum()
end

function Model:getOutputLabelsCount(batch)
  return self:getOutput(batch):ne(onmt.Constants.PAD):sum()
end

function Model:evaluate()
  for _, m in pairs(self.models) do
    m:evaluate()
  end
end

function Model:training()
  for _, m in pairs(self.models) do
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
    local p, gp = self.models[key]:getParameters()
    numParams = numParams + p:size(1)
    table.insert(params, p)
    table.insert(gradParams, gp)
  end

  if verbose then
    _G.logger:info(' * number of parameters: ' .. numParams)
  end

  return params, gradParams
end

return Model
