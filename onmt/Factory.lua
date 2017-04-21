local Factory = torch.class('Factory')

local options = {
  {
    '-brnn', false,
    [[Use a bidirectional encoder.]],
    {
      structural = 0
    }
  },
  {
    '-dbrnn', false,
    [[Use a deep bidirectional encoder.]],
    {
      structural = 0
    }
  },
  {
    '-pdbrnn', false,
    [[Use a pyramidal deep bidirectional encoder.]],
    {
      structural = 0
    }
  },
  {
    '-attention', 'global',
    [[Attention model.]],
    {
      enum = {'none', 'global'}
    }
  },
  {
    '-criterion', 'nll',
    [[Output criterion.]],
    {
      enum = {'nll', 'nce'},
      structural = 1
    }
  }
}

function Factory.declareOpts(cmd)
  cmd:setCmdLineOptions(options)
  onmt.BiEncoder.declareOpts(cmd)
  onmt.DBiEncoder.declareOpts(cmd)
  onmt.PDBiEncoder.declareOpts(cmd)
  onmt.GlobalAttention.declareOpts(cmd)
  onmt.NCEModule.declareOpts(cmd)
end

-- Return effective embeddings size based on user options.
local function resolveEmbSizes(opt, dicts, wordSizes)
  local wordEmbSize
  local featEmbSizes = {}

  wordSizes = onmt.utils.String.split(tostring(wordSizes), ',')

  if type(opt.word_vec_size) == 'number' and opt.word_vec_size > 0 then
    wordEmbSize = opt.word_vec_size
  else
    wordEmbSize = tonumber(wordSizes[1])
  end

  for i = 1, #dicts.features do
    local size

    if i + 1 <= #wordSizes then
      size = tonumber(wordSizes[i + 1])
    elseif opt.feat_merge == 'sum' then
      size = opt.feat_vec_size
    else
      size = math.floor(dicts.features[i]:size() ^ opt.feat_vec_exponent)
    end

    table.insert(featEmbSizes, size)
  end

  return wordEmbSize, featEmbSizes
end

local function buildInputNetwork(opt, dicts, wordSizes, pretrainedWords, fixWords, verbose)
  local wordEmbSize, featEmbSizes = resolveEmbSizes(opt, dicts, wordSizes)

  local wordEmbedding = onmt.WordEmbedding.new(dicts.words:size(), -- vocab size
                                               wordEmbSize,
                                               pretrainedWords,
                                               fixWords)

  local inputs
  local inputSize = wordEmbSize

  local multiInputs = #dicts.features > 0

  if multiInputs then
    inputs = nn.ParallelTable()
      :add(wordEmbedding)
  else
    inputs = wordEmbedding
  end

  if verbose then
    _G.logger:info('   - word embeddings size: ' .. wordEmbSize)
  end

  -- Sequence with features.
  if #dicts.features > 0 then
    if verbose then
      _G.logger:info('   - features embeddings sizes: ' .. table.concat(featEmbSizes, ', '))
    end

    local vocabSizes = {}
    for i = 1, #dicts.features do
      table.insert(vocabSizes, dicts.features[i]:size())
    end

    local featEmbedding = onmt.FeaturesEmbedding.new(vocabSizes, featEmbSizes, opt.feat_merge)
    inputs:add(featEmbedding)
    inputSize = inputSize + featEmbedding.outputSize
  end

  local inputNetwork

  if multiInputs then
    inputNetwork = nn.Sequential()
      :add(inputs)
      :add(nn.JoinTable(2, 2))
  else
    inputNetwork = inputs
  end

  inputNetwork.inputSize = inputSize

  return inputNetwork
end

function Factory.getOutputSizes(dicts)
  local outputSizes = { dicts.words:size() }
  for i = 1, #dicts.features do
    table.insert(outputSizes, dicts.features[i]:size())
  end
  return outputSizes
end

function Factory.buildEncoder(opt, inputNetwork, verbose)

  local function describeEncoder(name)
    if verbose then
      _G.logger:info('   - type: %s', name)
      _G.logger:info('   - structure: cell = %s; layers = %d; rnn_size = %d; dropout = ' .. opt.dropout,
                     opt.rnn_type, opt.layers, opt.rnn_size)
    end
  end

  if opt.brnn then
    describeEncoder('bidirectional')
    return onmt.BiEncoder.new(opt, inputNetwork)
  elseif opt.dbrnn then
    describeEncoder('deep bidirectional')
    return onmt.DBiEncoder.new(opt, inputNetwork)
  elseif opt.pdbrnn then
    describeEncoder('pyramidal deep bidirectional')
    return onmt.PDBiEncoder.new(opt, inputNetwork)
  else
    describeEncoder('simple')
    return onmt.Encoder.new(opt, inputNetwork)
  end

end

function Factory.buildWordEncoder(opt, dicts, verbose)
  if verbose then
    _G.logger:info(' * Encoder:')
  end

  local inputNetwork = buildInputNetwork(opt, dicts,
                                         opt.src_word_vec_size or opt.word_vec_size,
                                         opt.pre_word_vecs_enc, opt.fix_word_vecs_enc == 1,
                                         verbose)

  return Factory.buildEncoder(opt, inputNetwork, verbose)
end

function Factory.loadEncoder(pretrained, clone)
  if clone then
    pretrained = onmt.utils.Tensor.deepClone(pretrained)
  end

  local encoder

  if pretrained.name == 'Encoder' then
    encoder = onmt.Encoder.load(pretrained)
  elseif pretrained.name == 'BiEncoder' then
    encoder = onmt.BiEncoder.load(pretrained)
  elseif pretrained.name == 'PDBiEncoder' then
    encoder = onmt.PDBiEncoder.load(pretrained)
  elseif pretrained.name == 'DBiEncoder' then
    encoder = onmt.DBiEncoder.load(pretrained)
  else
    -- Keep for backward compatibility.
    local brnn = #pretrained.modules == 2
    if brnn then
      encoder = onmt.BiEncoder.load(pretrained)
    else
      encoder = onmt.Encoder.load(pretrained)
    end
  end

  return encoder
end

function Factory.buildDecoder(opt, inputNetwork, generator, attnModel, verbose)
  if verbose then
    _G.logger:info('   - structure: cell = %s; layers = %d; rnn_size = %d; dropout = ' .. opt.dropout,
                   opt.rnn_type, opt.layers, opt.rnn_size)
  end

  return onmt.Decoder.new(opt, inputNetwork, generator, attnModel)
end

function Factory.buildWordDecoder(opt, dicts, verbose)
  if verbose then
    _G.logger:info(' * Decoder:')
  end

  local inputNetwork = buildInputNetwork(opt, dicts,
                                         opt.tgt_word_vec_size or opt.word_vec_size,
                                         opt.pre_word_vecs_dec, opt.fix_word_vecs_dec == 1,
                                         verbose)

  local generator = Factory.buildGenerator(opt, dicts)
  local attnModel = Factory.buildAttention(opt)

  return Factory.buildDecoder(opt, inputNetwork, generator, attnModel, verbose)
end

function Factory.loadDecoder(pretrained, clone)
  if clone then
    pretrained = onmt.utils.Tensor.deepClone(pretrained)
  end

  local decoder = onmt.Decoder.load(pretrained)

  return decoder
end

function Factory.buildGenerator(opt, dicts)
  local sizes = Factory.getOutputSizes(dicts)
  return onmt.Generator(opt, dicts, sizes)
end

function Factory.buildCriterion(opt, dicts, verbose)
  if verbose then
    local modCriterion = ''
    if opt.criterion == 'nce' then
      modCriterion = ' (sample size: ' .. opt.nce_sample_size .. ')'
    end
    _G.logger:info(' * Criterion: '..opt.criterion..modCriterion)
  end

  local sizes = Factory.getOutputSizes(dicts)

  local criterion = nn.ParallelCriterion(false)
  criterion.normalizationFunc = {}

  for i = 1, #sizes do
    -- Ignores padding value.
    local w = torch.ones(sizes[i])
    w[onmt.Constants.PAD] = 0

    local feat_criterion
    if i == 1 and opt.criterion == 'nce' then
      feat_criterion = onmt.NCECriterion(w)
      table.insert(criterion.normalizationFunc, feat_criterion.normalize)
    else
      feat_criterion = nn.ClassNLLCriterion(w)
      -- Let the training code manage loss normalization.
      feat_criterion.sizeAverage = false
      -- no special normalization function
      table.insert(criterion.normalizationFunc, false)
    end
    criterion:add(feat_criterion)
  end

  return criterion
end

function Factory.buildAttention(args)
  if args.attention == 'none' then
    _G.logger:info('   - attention: none')
    return onmt.NoAttention(args, args.rnn_size)
  else
    _G.logger:info('   - attention: global (%s)', args.global_attention)
    return onmt.GlobalAttention(args, args.rnn_size)
  end
end

function Factory.loadGenerator(pretrained, clone)
  if clone then
    pretrained = onmt.utils.Tensor.deepClone(pretrained)
  end

  return pretrained
end

return Factory
