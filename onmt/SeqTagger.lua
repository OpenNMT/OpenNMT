--[[ Sequence to sequence model with attention. ]]
local SeqTagger, parent = torch.class('SeqTagger', 'Model')

local options = {
  {
    '-word_vec_size', '500',
    [[Comma-separated list of embedding sizes: word[,feat1,feat2,...].]],
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
    '-fix_word_vecs_enc', 0,
    [[Fix word embeddings on the encoder side.]],
    {
      enum = {0, 1},
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
    [[When features embedding sizes are not set and using -feat_merge concat, their dimension
      will be set to N^exponent where N is the number of values the feature takes.]],
    {
      structural = 0
    }
  },
  {
    '-feat_vec_size', 20,
    [[When features embedding sizes are not set and using -feat_merge sum,
      this is the common embedding size of the features]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt(),
      structural = 0
    }
  }
}


function SeqTagger.declareOpts(cmd)
  cmd:setCmdLineOptions(options, SeqTagger.modelName())
  onmt.Encoder.declareOpts(cmd)
  onmt.Factory.declareOpts(cmd)
end

function SeqTagger:__init(args, dicts, verbose)
  parent.__init(self, args)
  onmt.utils.Table.merge(self.args, onmt.utils.ExtendedCmdLine.getModuleOpts(args, options))

  self.models.encoder = onmt.Factory.buildWordEncoder(self.args, dicts.src, verbose)
  self.models.generator = onmt.Factory.buildGenerator(self.args.rnn_size, dicts.tgt)
  self.criterion = onmt.ParallelClassNLLCriterion(onmt.Factory.getOutputSizes(dicts.tgt))
end

function SeqTagger.load(args, models, dicts, isReplica)
  local self = torch.factory('SeqTagger')()

  parent.__init(self, args)
  onmt.utils.Table.merge(self.args, onmt.utils.ExtendedCmdLine.getModuleOpts(args, options))

  self.models.encoder = onmt.Factory.loadEncoder(models.encoder, isReplica)
  self.models.generator = onmt.Factory.loadGenerator(models.generator, isReplica)
  self.criterion = onmt.ParallelClassNLLCriterion(onmt.Factory.getOutputSizes(dicts.tgt))

  return self
end

-- Returns model name.
function SeqTagger.modelName()
  return 'Sequence Tagger'
end

-- Returns expected dataMode
function SeqTagger.dataType()
  return 'bitext'
end

function SeqTagger:enableProfiling()
  _G.profiler.addHook(self.models.encoder, 'encoder')
  _G.profiler.addHook(self.models.generator, 'generator')
  _G.profiler.addHook(self.criterion, 'criterion')
end

function SeqTagger:getOutput(batch)
  return batch.targetOutput
end

function SeqTagger:forwardComputeLoss(batch)
  local _, context = self.models.encoder:forward(batch)

  local loss = 0

  for t = 1, batch.sourceLength do
    local genOutputs = self.models.generator:forward(context:select(2, t))

    local output = batch:getTargetOutput(t)

    -- Same format with and without features.
    if torch.type(output) ~= 'table' then output = { output } end

    loss = loss + self.criterion:forward(genOutputs, output)
  end

  return loss
end

function SeqTagger:trainNetwork(batch)
  local loss = 0

  local _, context = self.models.encoder:forward(batch)

  local gradContexts = context:clone():zero()

  -- For each word of the sentence, generate target.
  for t = 1, batch.sourceLength do
    local genOutputs = self.models.generator:forward(context:select(2, t))

    local output = batch:getTargetOutput(t)

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

return SeqTagger
