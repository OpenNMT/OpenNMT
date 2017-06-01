--[[ Generic Model class. ]]
local Model = torch.class('Model')

local options = {
  {
    '-model_type', 'seq2seq',
    [[Type of model to train. This option impacts all options choices.]],
    {
      enum = {'lm', 'seq2seq', 'seqtagger'},
      structural = 0
    }
  },
  {
    '-param_init', 0.1,
    [[Parameters are initialized over uniform distribution with support (-`param_init`, `param_init`).]],
    {
      valid = onmt.utils.ExtendedCmdLine.isFloat(0),
      init_only = true
    }
  }
}

function Model.declareOpts(cmd)
  cmd:setCmdLineOptions(options, 'Model')
end

function Model:__init(args)
  self.args = onmt.utils.ExtendedCmdLine.getModuleOpts(args, options)
  self.models = {}
end

-- Dynamically change parameters in the graph.
function Model:changeParameters(changes)
  _G.logger:info('Applying new parameters:')

  for k, v in pairs(changes) do
    _G.logger:info(' * %s = ' .. tostring(v), k)

    for _, model in pairs(self.models) do
      model:apply(function(m)
        if k == 'dropout' and torch.typename(m) == 'nn.Dropout' then
          m:setp(v)
        elseif k:find('fix_word_vecs') and torch.typename(m) == 'onmt.WordEmbedding' then
          local enc = k == 'fix_word_vecs_enc' and torch.typename(model):find('Encoder')
          local dec = k == 'fix_word_vecs_dec' and torch.typename(model):find('Decoder')
          if enc or dec then
            m:fixEmbeddings(v)
          end
        end
      end)
    end
  end

end

function Model:dumpGraphs(path)
  for name, desc in pairs(self.models) do
    local net = desc.network or desc
    if net.fg then
      _G.logger:info('Generate graph '..name..'.dot')
      local MG=onmt.utils.MemoryGraph.new(net.fg)
      MG:dump(path..'/'..name..'.dot')
    end
  end
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

function Model:initParams()
  _G.logger:info('Initializing parameters...')

  local params, gradParams, modelMap = self:getParams()
  local numParams = 0

  for i = 1, #params do
    local name = modelMap[i]
    params[i]:uniform(-self.args.param_init, self.args.param_init)

    self.models[name]:apply(function (m)
      if m.postParametersInitialization then
        m:postParametersInitialization()
      end
    end)

    numParams = numParams + params[i]:size(1)
  end

  _G.logger:info(' * number of parameters: ' .. numParams)

  return params, gradParams
end

function Model:getParams()
  -- Order the model table because we need all replicas to have the same order.
  local orderedIndex = {}
  for key in pairs(self.models) do
    table.insert(orderedIndex, key)
  end
  table.sort(orderedIndex)

  local params = {}
  local gradParams = {}
  local modelMap = {}

  for _, key in ipairs(orderedIndex) do
    local p, gp = self.models[key]:getParameters()
    if p:dim() > 0 then
      table.insert(params, p)
      table.insert(gradParams, gp)
      table.insert(modelMap, key)
    end
  end

  return params, gradParams, modelMap
end

return Model
