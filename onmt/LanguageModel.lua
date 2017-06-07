--[[ Language Model. ]]
local LanguageModel, parent = torch.class('LanguageModel', 'Model')

local options = {
  {
    '-word_vec_size', { 500 },
    [[List of embedding sizes: `word[ feat1[ feat2[ ...] ] ]`.]],
    {
      structural = 0
    }
  },
  {
    '-pre_word_vecs_enc', '',
    [[Path to pretrained word embeddings on the encoder side serialized as a Torch tensor.]],
    {
      valid = onmt.utils.ExtendedCmdLine.fileNullOrExists,
      init_only = true
    }
  },
  {
    '-fix_word_vecs_enc', false,
    [[Fix word embeddings on the encoder side.]],
    {
      enum = { false, true, 'pretrained' },
      structural = 1
    }
  },
  {
    '-feat_merge', 'concat',
    [[Merge action for the features embeddings.]],
    {
      enum = {'concat', 'sum'},
      structural = 0
    }
  },
  {
    '-feat_vec_exponent', 0.7,
    [[When features embedding sizes are not set and using `-feat_merge concat`, their dimension
      will be set to `N^feat_vec_exponent` where `N` is the number of values the feature takes.]],
    {
      structural = 0
    }
  },
  {
    '-feat_vec_size', 20,
    [[When features embedding sizes are not set and using `-feat_merge sum`,
      this is the common embedding size of the features]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt(),
      structural = 0
    }
  }
}

function LanguageModel.declareOpts(cmd)
  cmd:setCmdLineOptions(options, 'Language Model')
  onmt.Encoder.declareOpts(cmd)
  onmt.Factory.declareOpts(cmd)
end

function LanguageModel:__init(args, dicts)
  parent.__init(self, args)
  onmt.utils.Table.merge(self.args, onmt.utils.ExtendedCmdLine.getModuleOpts(args, options))

  self.models.encoder = onmt.Factory.buildWordEncoder(args, dicts.src)
  self.models.generator = onmt.Factory.buildGenerator(args, dicts.src)

  self.criterion = onmt.ParallelClassNLLCriterion(onmt.Factory.getOutputSizes(dicts.src))

  self.eosProto = {}
  for _ = 1, #dicts.src.features + 1 do
    table.insert(self.eosProto, torch.LongTensor())
  end
end

function LanguageModel.load(args, models, dicts)
  local self = torch.factory('LanguageModel')()

  parent.__init(self, args)
  onmt.utils.Table.merge(self.args, onmt.utils.ExtendedCmdLine.getModuleOpts(args, options))

  self.models.encoder = onmt.Factory.loadEncoder(models.encoder)
  self.models.generator = onmt.Generator.load(models.generator)
  self.criterion = onmt.ParallelClassNLLCriterion(onmt.Factory.getOutputSizes(dicts.src))

  self.eosProto = {}
  for _ = 1, #dicts.src.features + 1 do
    table.insert(self.eosProto, torch.LongTensor())
  end

  return self
end

-- Returns model name.
function LanguageModel.modelName()
  return 'Language'
end

-- Returns expected default datatype or if passed a parameter, returns if it is supported
function LanguageModel.dataType(datatype)
  if not datatype then
    return 'monotext'
  else
    return datatype == 'monotext'
  end
end

function LanguageModel:enableProfiling()
  _G.profiler.addHook(self.models.encoder, 'encoder')
  _G.profiler.addHook(self.models.generator, 'generator')
  _G.profiler.addHook(self.criterion, 'criterion')
end

function LanguageModel:getOutput(batch)
  return batch.sourceInput
end

function LanguageModel:forwardComputeLoss(batch, indvLoss)
  local _, context = self.models.encoder:forward(batch)
  local eos = onmt.utils.Tensor.reuseTensorTable(self.eosProto, { batch.size })
  for i = 1, #eos do
    eos[i]:fill(onmt.Constants.EOS)
  end

  local loss = 0

  local indvAvgLoss = torch.zeros(batch.size)

  for t = 1, batch.sourceLength do
    local genOutputs = self.models.generator:forward(context:select(2, t))

    -- LanguageModel is supposed to predict the following word.
    local output
    if t ~= batch.sourceLength then
      output = batch:getSourceInput(t + 1)
    else
      output = eos
    end

    -- Same format with and without features.
    if torch.type(output) ~= 'table' then output = { output } end

    if indvLoss then
      for i = 1, batch.size do
        local tmpPred = {}
        local tmpOutput = {}
        for j = 1, #genOutputs do
          table.insert(tmpPred, genOutputs[j][{{i}, {}}])
          table.insert(tmpOutput, output[j][{{i}}])
        end
        local tmpLoss = self.criterion:forward(tmpPred, tmpOutput)
        indvAvgLoss[i] = indvAvgLoss[i] + tmpLoss
        loss = loss + tmpLoss
      end
    else
      loss = loss + self.criterion:forward(genOutputs, output)
    end
  end

  if indvLoss then
    indvAvgLoss = torch.cdiv(indvAvgLoss, batch.sourceSize:double())
  end

  return loss, indvAvgLoss
end

function LanguageModel:trainNetwork(batch)
  local loss = 0

  local _, context = self.models.encoder:forward(batch)

  local gradContexts = context:clone():zero()
  local eos = onmt.utils.Tensor.reuseTensorTable(self.eosProto, { batch.size })
  for i = 1, #eos do
    eos[i]:fill(onmt.Constants.EOS)
  end

  -- For each word of the sentence, generate target.
  for t = 1, batch.sourceLength do
    local genOutputs = self.models.generator:forward(context:select(2, t))

    -- LanguageModel is supposed to predict following word.
    local output
    if t ~= batch.sourceLength then
      output = batch:getSourceInput(t + 1)
    else
      output = eos
    end

    -- Same format with and without features.
    if torch.type(output) ~= 'table' then output = { output } end

    loss = loss + self.criterion:forward(genOutputs, output)

    local genGradOutput = self.criterion:backward(genOutputs, output)
    for j = 1, #genGradOutput do
      genGradOutput[j]:div(batch.totalSize)
    end

    gradContexts[{{}, t}]:copy(self.models.generator:backward(context:select(2, t), genGradOutput))
  end

  self.models.encoder:backward(batch, nil, gradContexts)

  return loss
end

return LanguageModel
