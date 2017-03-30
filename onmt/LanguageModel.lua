--[[ Language Model. ]]
local LanguageModel, parent = torch.class('LanguageModel', 'Model')

local options = {
  {'-word_vec_size', '500', [[Comma-separated list of embedding sizes: word[,feat1,feat2,...].]]},
  {'-pre_word_vecs_enc', '', [[If a valid path is specified, then this will load
                                     pretrained word embeddings on the encoder side.
                                     See README for specific formatting instructions.]],
                         {valid=onmt.utils.ExtendedCmdLine.fileNullOrExists}},
  {'-fix_word_vecs_enc', 0, [[Fix word embeddings on the encoder side]]},
  {'-feat_merge', 'concat', [[Merge action for the features embeddings.]],
                     {enum={'concat', 'sum'}}},
  {'-feat_vec_exponent', 0.7, [[When using concatenation, if the feature takes N values
                                        then the embedding dimension will be set to N^exponent]]},
  {'-feat_vec_size', 20, [[When using sum, the common embedding size of the features]]}
}

function LanguageModel.declareOpts(cmd)
  cmd:setCmdLineOptions(options, 'Language Model')
  onmt.Encoder.declareOpts(cmd)
  onmt.Factory.declareOpts(cmd)
end

function LanguageModel:__init(args, dicts, verbose)
  parent.__init(self, args)
  onmt.utils.Table.merge(self.args, onmt.utils.ExtendedCmdLine.getModuleOpts(args, options))

  self.models.encoder = onmt.Factory.buildWordEncoder(self.args, dicts.src, verbose)
  self.models.generator = onmt.Factory.buildGenerator(self.args.rnn_size, dicts.src)

  self.criterion = onmt.ParallelClassNLLCriterion(onmt.Factory.getOutputSizes(dicts.src))

  self.eosProto = {}
  for _ = 1, #dicts.src.features + 1 do
    table.insert(self.eosProto, torch.LongTensor())
  end
end

function LanguageModel.load(args, models, dicts, isReplica)
  local self = torch.factory('LanguageModel')()

  parent.__init(self, args)
  onmt.utils.Table.merge(self.args, onmt.utils.ExtendedCmdLine.getModuleOpts(args, options))

  self.models.encoder = onmt.Factory.loadEncoder(models.encoder, isReplica)
  self.models.generator = onmt.Factory.loadGenerator(models.generator, isReplica)
  self.criterion = onmt.ParallelClassNLLCriterion(onmt.Factory.getOutputSizes(dicts.src))

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
