--[[ seq2seq Model. ]]
local seq2seq, parent = torch.class('onmt.Models.lm', 'onmt.Model')

function seq2seq:__init(opt, datasetOrCheckpoint, verboseOrReplica)
  parent.__init(self)
  if type(datasetOrCheckpoint)=='Checkpoint' then
    local checkpoint = datasetOrCheckpoint
    local dataset = checkpoint.dataset
    local replica = verbose
    self.models.encoder = onmt.Models.loadEncoder(checkpoint.models, replica)
    self.models.generator = onmt.Models.loadGenerator(checkpoint.models, replica)
  else
    local dataset = datasetOrCheckpoint
    local verbose = verboseOrReplica
    self.models.encoder = onmt.Models.buildEncoder(opt, dataset.dicts)
    if #data.dicts.features > 0 then
      self.models.generator = onmt.FeaturesGenerator.new(opt.rnn_size, data.dicts.words:size(), data.dicts.features)
    else
      self.models.generator = onmt.Generator.new(opt.rnn_size, dataset.dicts.words:size())
    end
  end
  self.EOS_vector_model = torch.LongTensor(opt.max_batch_size):fill(dataset.dicts.words:lookup(onmt.Constants.EOS_WORD))
end


function seq2seq:forwardComputeLoss(batch, criterion)
  for i = 1, data:batchCount() do
    local batch = onmt.utils.Cuda.convert(data:getBatch(i))
    local _, context = self.models.encoder:forward(batch)
    local EOS_vector = self.EOS_vector_model:narrow(1, 1, batch.size)
    onmt.utils.Cuda.convert(EOS_vector)

    for t = 1, batch.sourceLength do
      local genOutputs = self.models.generator:forward(context:select(2, t))
      -- LM is supposed to predict the following word.
      local output
      if t ~= batch.sourceLength then
        output = batch:getSourceInput(t + 1)
      else
        output = EOS_vector
      end
      -- Same format with and without features.
      if torch.type(output) ~= 'table' then output = { output } end
      loss = loss + criterion:forward(genOutputs, output)
    end
  end
  return loss
end

function seq2seq:buildCriterion(dataset)
  return onmt.Criterion.new(dataset.dicts.src.words:size(),
                            dataset.dicts.src.features)
end

function seq2seq:trainNetwork(batch, gradParams, criterion, doProfile, dryRun)
  local loss = 0
  local _, context = self.models.encoder:forward(batch)
  local gradContexts = torch.Tensor(batch.size, batch.sourceLength, opt.rnn_size)
  gradContexts = onmt.utils.Cuda.convert(gradContexts)
  -- for each word of the sentence, generate target
  for t = 1, batch.sourceLength do
    local genOutputs = self.models.generator:forward(context:select(2,t))
    -- LM is supposed to predict following word
    local output
    if t ~= batch.sourceLength then
      output = batch:getSourceInput(t + 1)
    else
      output = self.EOS_vector_model:narrow(1, 1, batch.size)
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

function seq2seq.declareOpts(cmd)
  cmd:option('-layers', 2, [[Number of layers in the RNN encoder/decoder]])
  cmd:option('-rnn_size', 500, [[Size of RNN hidden states]])
  cmd:option('-rnn_type', 'LSTM', [[Type of RNN cell: LSTM, GRU]])
  cmd:option('-word_vec_size', 500, [[Word embedding sizes]])
  cmd:option('-feat_merge', 'concat', [[Merge action for the features embeddings: concat or sum]])
  cmd:option('-feat_vec_exponent', 0.7, [[When using concatenation, if the feature takes N values
                                        then the embedding dimension will be set to N^exponent]])
  cmd:option('-feat_vec_size', 20, [[When using sum, the common embedding size of the features]])
  cmd:option('-input_feed', 1, [[Feed the context vector at each time step as additional input (via concatenation with the word embeddings) to the decoder.]])
  cmd:option('-residual', false, [[Add residual connections between RNN layers.]])
  cmd:option('-brnn', false, [[Use a bidirectional encoder]])
  cmd:option('-brnn_merge', 'sum', [[Merge action for the bidirectional hidden states: concat or sum]])
  cmd:option('-pre_word_vecs_enc', '', [[If a valid path is specified, then this will load
                                     pretrained word embeddings on the encoder side.
                                     See README for specific formatting instructions.]])
  cmd:option('-fix_word_vecs_enc', false, [[Fix word embeddings on the encoder side]])

end
