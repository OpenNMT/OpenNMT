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
  self.criterion = onmt.Factory.buildCriterion(args, dicts.src)

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
  self.models.generator = onmt.Factory.loadGenerator(models.generator)
  self.criterion = onmt.Factory.buildCriterion(args, dicts.src)

  return self
end

-- Returns model name.
function LanguageModel.modelName()
  return 'Language'
end

-- Returns expected dataMode.
function LanguageModel.dataType()
  return 'monotext'
end

function LanguageModel:enableProfiling()
  _G.profiler.addHook(self.models.encoder, 'encoder')
  _G.profiler.addHook(self.models.generator, 'generator')
  _G.profiler.addHook(self.criterion, 'criterion')
end

function LanguageModel:getOutput(batch)
  return batch.sourceInput
end

function LanguageModel:forwardComputeLoss(batch)
  local _, context = self.models.encoder:forward(batch)
  local eos = onmt.utils.Tensor.reuseTensorTable(self.eosProto, { batch.size })
  for i = 1, #eos do
    eos[i]:fill(onmt.Constants.EOS)
  end

  local loss = 0

  for t = 1, batch.sourceLength do
    -- LanguageModel is supposed to predict the following word.
    local output
    if t ~= batch.sourceLength then
      output = batch:getSourceInput(t + 1)
    else
      output = eos
    end

    local prepOutputs = context:select(2, t)
    -- sampling-based generator need outputs during training
    -- use generator specific flag to keep backward compatibility
    if self.models.generator.needOutput then
      if type(output) == 'table' then
        prepOutputs = { prepOutputs, output }
      else
        prepOutputs = { prepOutputs, { output } }
      end
    end

    local genOutputs = self.models.generator:forward(prepOutputs)

    -- Same format with and without features.
    if torch.type(output) ~= 'table' then output = { output } end

    loss = loss + self.criterion:forward(genOutputs, output)
  end

  return loss
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
    -- LanguageModel is supposed to predict following word.
    local output
    if t ~= batch.sourceLength then
      output = batch:getSourceInput(t + 1)
    else
      output = eos
    end

    local prepOutputs = context:select(2, t)
    -- sampling-based generator need outputs during training
    -- use generator specific flag to keep backward compatibility
    if self.models.generator.needOutput then
      if type(output) == 'table' then
        prepOutputs = { prepOutputs, output }
      else
        prepOutputs = { prepOutputs, { output } }
      end
    end

    local genOutputs = self.models.generator:forward(prepOutputs)

    -- Same format with and without features.
    if torch.type(output) ~= 'table' then output = { output } end

    loss = loss + self.criterion:forward(genOutputs, output)

    local genGradOutput = self.criterion:backward(genOutputs, output)

    -- normalize gradient - we might have several batches in parallel, so we divide by total size of batch
    for j = 1, #genGradOutput do
      -- each criterion might have its own normalization function
      if self.criterion.normalizationFunc and self.criterion.normalizationFunc[j] ~= false then
        self.criterion.normalizationFunc[j](genGradOutput[j], batch.totalSize)
      else
        genGradOutput[j]:div(batch.totalSize)
      end
    end

    local decGradOut = self.models.generator:backward(prepOutputs, genGradOutput)

    -- if we sent the output, then get gradient back on the input
    if self.models.generator.needOutput then
      decGradOut = decGradOut[1]
    end

    gradContexts[{{}, t}]:copy(decGradOut)
  end

  self.models.encoder:backward(batch, nil, gradContexts)

  return loss
end

return LanguageModel
