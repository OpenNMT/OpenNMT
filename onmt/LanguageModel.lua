--[[ Language Model. ]]
local LanguageModel, parent = torch.class('LanguageModel', 'Model')

local LanguageModel_options = {
  {'-layers', 2, [[Number of layers in the RNN encoder/decoder]]},
  {'-rnn_size', 500, [[Size of RNN hidden states]]},
  {'-rnn_type', 'LSTM', [[Type of RNN cell: LSTM, GRU]]},
  {'-word_vec_size', '500', [[Comma-separated list of embedding sizes: word[,feat1,feat2,...].]]},
  {'-feat_merge', 'concat', [[Merge action for the features embeddings.]],
                     {enum={'concat','sum'}}},
  {'-feat_vec_exponent', 0.7, [[When using concatenation, if the feature takes N values
                                        then the embedding dimension will be set to N^exponent]]},
  {'-feat_vec_size', 20, [[When using sum, the common embedding size of the features]]},
  {'-residual', false, [[Add residual connections between RNN layers.]]},
  {'-brnn', false, [[Use a bidirectional encoder]]},
  {'-brnn_merge', 'sum', [[Merge action for the bidirectional hidden states.]],
                     {enum={'concat','sum'}}},
  {'-pre_word_vecs_enc', '', [[If a valid path is specified, then this will load
                                     pretrained word embeddings on the encoder side.
                                     See README for specific formatting instructions.]],
                         {valid=onmt.utils.ExtendedCmdLine.fileNullOrExists}},
  {'-fix_word_vecs_enc', false, [[Fix word embeddings on the encoder side]]},
  {'-dropout', 0.3, [[Dropout probability. Dropout is applied between vertical LSTM stacks.]]}
}

function LanguageModel.declareOpts(cmd)
  cmd:setCmdLineOptions(LanguageModel_options, "Language Model")
end

function LanguageModel:__init(args, dicts)
  parent.__init(self, args)
  onmt.utils.Table.merge(self.args, onmt.utils.ExtendedCmdLine.getModuleOpts(args, LanguageModel_options))

  self.models.encoder = onmt.Factory.buildWordEncoder(self.args, dicts.src)

  if #dicts.src.features > 0 then
    self.models.generator = onmt.FeaturesGenerator.new(self.args.rnn_size,
                                                       dicts.src.words:size(),
                                                       dicts.src.features)
  else
    self.models.generator = onmt.Generator.new(self.args.rnn_size, dicts.src.words:size())
  end

  self.eosProto = {}
  for _ = 1, #dicts.src.features + 1 do
    table.insert(self.eosProto, torch.LongTensor())
  end
end

function LanguageModel.load()
  error("loading a language model is not yet supported")
end

-- Returns model name.
function LanguageModel.modelName()
  return "Language"
end

-- Returns expected dataMode.
function LanguageModel.dataType()
  return "monotext"
end

function LanguageModel:getOutput(batch)
  return batch.sourceInput
end

function LanguageModel:forwardComputeLoss(batch, criterion)
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
    loss = loss + criterion:forward(genOutputs, output)
  end
  return loss
end

function LanguageModel:buildCriterion(dicts)
  local outputSizes = { dicts.src.words:size() }
  for j = 1, #dicts.src.features do
    table.insert(outputSizes, dicts.src.features[j]:size())
  end

  return onmt.ParallelClassNLLCriterion(outputSizes)
end

function LanguageModel:trainNetwork(batch, criterion)
  local loss = 0

  local _, context = self.models.encoder:forward(batch)

  local gradContexts = context:clone():zero()
  local eos = onmt.utils.Tensor.reuseTensorTable(self.eosProto, { batch.size })
  for i = 1, #eos do
    eos[i]:fill(onmt.Constants.EOS)
  end

  -- for each word of the sentence, generate target
  for t = 1, batch.sourceLength do
    local genOutputs = self.models.generator:forward(context:select(2,t))

    -- LanguageModel is supposed to predict following word
    local output
    if t ~= batch.sourceLength then
      output = batch:getSourceInput(t + 1)
    else
      output = eos
    end
    -- same format with and without features
    if torch.type(output) ~= 'table' then output = { output } end

    loss = loss + criterion:forward(genOutputs, output)

    -- backward
    local genGradOutput = criterion:backward(genOutputs, output)
    for j = 1, #genGradOutput do
      genGradOutput[j]:div(batch.totalSize)
    end

    gradContexts[{{}, t}]:copy(self.models.generator:backward(context:select(2, t), genGradOutput))
  end

  self.models.encoder:backward(batch, nil, gradContexts)

  return loss
end

return LanguageModel
