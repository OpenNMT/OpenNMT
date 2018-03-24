--[[ Sequence tagger. ]]
local SeqTagger, parent = torch.class('SeqTagger', 'Model')

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
  },
  {
    '-loglikelihood', 'word',
    [[Specifies the type of loglikelihood of the tagger model;
    'word' indicates tags are predicted at the word-level, and
    'sentence' indicates tagging process is treated as a markov chain]],
    {
      enum = {'word', 'sentence'},
      structural = 1
    }
  }
}


function SeqTagger.declareOpts(cmd)
  cmd:setCmdLineOptions(options, SeqTagger.modelName())
  onmt.Encoder.declareOpts(cmd)
  onmt.Factory.declareOpts(cmd)
end

function SeqTagger:__init(args, dicts)
  parent.__init(self, args)
  onmt.utils.Table.merge(self.args, onmt.utils.ExtendedCmdLine.getModuleOpts(args, options))

  if not dicts.src then
    -- the input is already a vector
    args.dimInputSize = dicts.srcInputSize
  end

  self.models.encoder = onmt.Factory.buildWordEncoder(args, dicts.src)
  self.models.generator = onmt.Factory.buildGenerator(args, dicts.tgt)

  onmt.utils.Error.assert(args.loglikelihood == 'word' or args.loglikelihood == 'sentence',
    'Invalid loglikelihood type of SeqTagger `%s\'', args.loglikelihood)

  self.loglikelihood = args.loglikelihood

  if self.loglikelihood == 'word' then
    self.criterion = onmt.ParallelCriterion(onmt.Factory.getOutputSizes(dicts.tgt))
  elseif self.loglikelihood == 'sentence' then
    self.criterion = onmt.SentenceNLLCriterion(args, onmt.Factory.getOutputSizes(dicts.tgt))
    self.models.criterion = self.criterion -- criterion is model parameter
  end
end

function SeqTagger.load(args, models, dicts)
  local self = torch.factory('SeqTagger')()

  parent.__init(self, args)
  onmt.utils.Table.merge(self.args, onmt.utils.ExtendedCmdLine.getModuleOpts(args, options))

  self.models.encoder = onmt.Factory.loadEncoder(models.encoder)
  self.models.generator = onmt.Factory.loadGenerator(models.generator)

  onmt.utils.Error.assert(args.loglikelihood == 'word' or args.loglikelihood == 'sentence',
                                  'Invalid loglikelihood type of SeqTagger `%s\'', args.loglikelihood)

  if args.loglikelihood == 'word' then
    self.criterion = onmt.ParallelCriterion(onmt.Factory.getOutputSizes(dicts.tgt))
    self.loglikelihood = 'word'
  elseif args.loglikelihood == 'sentence' then
    if not models.criterion then -- loading pre-trained word model to further train sentence model
      _G.logger:info('Creating a new SentenceNLLCriterion')
      self.criterion = onmt.SentenceNLLCriterion(args, onmt.Factory.getOutputSizes(dicts.tgt))
      local p, g = self.criterion:getParameters()
      p:uniform(-args.param_init, args.param_init)
      g:uniform(-args.param_init, args.param_init)
      self.criterion:postParametersInitialization()
    else
      self.criterion = onmt.Factory.loadSentenceNLLCriterion(models.criterion)
    end
    self.models.criterion = self.criterion
    self.loglikelihood = 'sentence'
  end

  return self
end

-- Returns model name.
function SeqTagger.modelName()
  return 'Sequence Tagger'
end

-- Returns expected default datatype or if passed a parameter, returns if it is supported
function SeqTagger.dataType(datatype)
  if not datatype then
    return 'bitext'
  else
    return datatype == 'bitext' or datatype == 'feattext'
  end
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

  if self.loglikelihood == 'sentence' then
    local reference = batch.targetOutput:t() -- SeqLen x B  -> B x SeqLen
    local tagsScoreTable = {}
    for t = 1, batch.sourceLength do
      local tagsScore = self.models.generator:forward(context:select(2, t)) -- B x TagSize
      -- tagsScore is a table
      tagsScore = nn.utils.addSingletonDimension(tagsScore[1], 3):clone() -- B x TagSize x 1
      table.insert(tagsScoreTable, tagsScore)
    end
    local tagsScores = nn.JoinTable(3):forward(tagsScoreTable)  -- B x TagSize x SeqLen
    loss = self.models.criterion:forward(tagsScores, reference)
  else -- 'word'
    for t = 1, batch.sourceLength do
      local genOutputs = self.models.generator:forward(context:select(2, t))

      local output = batch:getTargetOutput(t)

      -- Same format with and without features.
      if torch.type(output) ~= 'table' then output = { output } end

      loss = loss + self.criterion:forward(genOutputs, output)
    end
  end

  return loss
end

function SeqTagger:trainNetwork(batch)
  local loss = 0

  local _, context = self.models.encoder:forward(batch)

  local gradContexts = context:clone():zero()

  if self.loglikelihood == 'sentence' then

    local reference = batch.targetOutput:t() -- SeqLen x B  -> B x SeqLen
    local B = batch.size
    local T = batch.sourceLength

    local tagsScoreTable = {}
    for t = 1, T do
      local tagsScore = self.models.generator:forward(context:select(2, t)) -- B x TagSize
      -- tagsScore is a table
      tagsScore = nn.utils.addSingletonDimension(tagsScore[1], 3):clone() -- B x TagSize x 1
      table.insert(tagsScoreTable, tagsScore)
    end
    local tagsScores = nn.JoinTable(3):forward(tagsScoreTable) -- B x TagSize x SeqLen

    loss = loss + self.criterion:forward(tagsScores, reference)

    local gradCriterion = self.models.criterion:backward(tagsScores, reference) -- B x TagSize x SeqLen

    gradCriterion = torch.div(gradCriterion, B)
    for t = 1, T do
      gradContexts:select(2,t):copy(self.models.generator:backward(context:select(2, t), {gradCriterion:select(3,t)}))
    end

  else -- 'word'
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
  end

  self.models.encoder:backward(batch, nil, gradContexts)

  return loss
end

function SeqTagger:tagBatch(batch)
  local pred = {}
  local feats = {}

  for _ = 1, batch.size do
    table.insert(pred, {})
    table.insert(feats, {})
  end
  local _, context = self.models.encoder:forward(batch)

  if self.loglikelihood == 'sentence' then

    local tagsScoreTable = {}
    for t = 1, batch.sourceLength do
      local tagsScore = self.models.generator:forward(context:select(2, t)) -- B x TagSize
      -- tagsScore is a table
      tagsScore = nn.utils.addSingletonDimension(tagsScore[1], 3):clone() -- B x TagSize x 1
      table.insert(tagsScoreTable, tagsScore)
    end
    local tagsScores = onmt.utils.Cuda.convert(nn.JoinTable(3):forward(tagsScoreTable))  -- B x TagSize x SeqLen

    -- viterbi search
    local senPreds = self.criterion:viterbiSearch(tagsScores, batch.sourceSize) -- B x SeqLen (type Long)

    for t = 1, batch.sourceLength do
      for b = 1, batch.size do
        -- padded in the beginning
        if t > batch.sourceLength - batch.sourceSize[b] then
          pred[b][t - batch.sourceLength + batch.sourceSize[b]] = senPreds[b][t]
          feats[b][t - batch.sourceLength + batch.sourceSize[b]] = {}
        end
      end
    end

  else -- 'word'
    for t = 1, batch.sourceLength do
      local out = self.models.generator:forward(context:select(2, t))
      if type(out[1]) == 'table' then
        out = out[1]
      end
      local _, best = out[1]:max(2)
      for b = 1, batch.size do
        if t > batch.sourceLength - batch.sourceSize[b] then
          pred[b][t - batch.sourceLength + batch.sourceSize[b]] = best[b][1]
          feats[b][t - batch.sourceLength + batch.sourceSize[b]] = {}
        end
      end
      for j = 2, #out do
        _, best = out[j]:max(2)
        for b = 1, batch.size do
          if t > batch.sourceLength - batch.sourceSize[b] then
            feats[b][t - batch.sourceLength + batch.sourceSize[b]][j - 1] = best[b][1]
          end
        end
      end
    end

  end

  return pred, feats
end

return SeqTagger
