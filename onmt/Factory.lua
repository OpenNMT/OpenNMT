local Factory = torch.class('Factory')

local options = {
  {
    '-encoder_type', 'rnn',
    [[Encoder type.]],
    {
      enum = { 'rnn', 'brnn', 'dbrnn', 'pdbrnn', 'gnmt', 'cnn' },
      structural = 0,
      depends = function(opt)
                  if opt.encoder_type == 'cnn' then
                    if opt.bridge == 'copy' then
                      return false, "CNN encoder doesn't work with copy bridge. Please use either 'none' or 'dense'." end
                  end
                  return true
                end
    }
  },
  {
    '-brnn', false,
    [[Use a bidirectional encoder.]],
    {
      deprecatedBy = { 'encoder_type', 'brnn' },
      structural = 0
    }
  },
  {
    '-dbrnn', false,
    [[Use a deep bidirectional encoder.]],
    {
      deprecatedBy = { 'encoder_type', 'dbrnn' },
      structural = 0
    }
  },
  {
    '-pdbrnn', false,
    [[Use a pyramidal deep bidirectional encoder.]],
    {
      deprecatedBy = { 'encoder_type', 'pdbrnn' },
      structural = 0
    }
  },
  {
    '-attention', 'global',
    [[Attention model.]],
    {
      enum = {'none', 'global'},
      structural = 0
    }
  }
}

function Factory.declareOpts(cmd)
  cmd:setCmdLineOptions(options)
  onmt.BiEncoder.declareOpts(cmd)
  onmt.DBiEncoder.declareOpts(cmd)
  onmt.PDBiEncoder.declareOpts(cmd)
  onmt.GoogleEncoder.declareOpts(cmd)
  onmt.CNNEncoder.declareOpts(cmd)
  onmt.GlobalAttention.declareOpts(cmd)
end

-- Return effective embeddings size based on user options.
local function resolveEmbSizes(opt, dicts, wordSizes)
  local wordEmbSize
  local featEmbSizes = {}

  if type(wordSizes) ~= 'table' then
    wordSizes = { wordSizes }
  end

  if type(opt.word_vec_size) == 'number' and opt.word_vec_size > 0 then
    wordEmbSize = opt.word_vec_size
  else
    wordEmbSize = wordSizes[1]
  end

  for i = 1, #dicts.features do
    local size

    if i + 1 <= #wordSizes then
      size = wordSizes[i + 1]
    elseif opt.feat_merge == 'sum' then
      size = opt.feat_vec_size
    else
      size = math.floor(dicts.features[i]:size() ^ opt.feat_vec_exponent)
    end

    table.insert(featEmbSizes, size)
  end

  return wordEmbSize, featEmbSizes
end

local function buildInputNetwork(opt, dicts, wordSizes, pretrainedWords, fixWords)

  if not dicts then
    -- if input vector - skip word embbedding
    local inputNetwork = nn.Identity()
    inputNetwork.inputSize = opt.dimInputSize
    return inputNetwork
  end

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

  _G.logger:info('   - word embeddings size: ' .. wordEmbSize)

  -- Sequence with features.
  if #dicts.features > 0 then
    _G.logger:info('   - features embeddings sizes: ' .. table.concat(featEmbSizes, ', '))

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

local function describeRNN(opt)
  _G.logger:info('   - structure: cell = %s; layers = %d; rnn_size = %d; dropout = %0.2f (%s)',
                 opt.rnn_type, opt.layers, opt.rnn_size, opt.dropout, opt.dropout_type)
end

local function describeCNN(opt)
  _G.logger:info('   - structure: cnn_kernel = %d; cnn_layers = %d; cnn_size = %d;',
                   opt.cnn_kernel, opt.cnn_layers, opt.cnn_size)
end

function Factory.getOutputSizes(dicts)
  local outputSizes = { dicts.words:size() }
  for i = 1, #dicts.features do
    table.insert(outputSizes, dicts.features[i]:size())
  end
  return outputSizes
end

function Factory.buildEncoder(opt, inputNetwork)

  local function describeEncoder(name)
    _G.logger:info('   - type: %s', name)
    if name == 'CNN' then
      describeCNN(opt)
    else
      describeRNN(opt)
    end
  end

  if opt.encoder_type == 'brnn' then
    describeEncoder('bidirectional RNN')
    return onmt.BiEncoder.new(opt, inputNetwork)
  elseif opt.encoder_type == 'dbrnn' then
    describeEncoder('deep bidirectional RNN')
    return onmt.DBiEncoder.new(opt, inputNetwork)
  elseif opt.encoder_type == 'pdbrnn' then
    describeEncoder('pyramidal deep bidirectional RNN')
    return onmt.PDBiEncoder.new(opt, inputNetwork)
  elseif opt.encoder_type == 'gnmt' then
    describeEncoder('GNMT')
    return onmt.GoogleEncoder.new(opt, inputNetwork)
  elseif opt.encoder_type == 'cnn' then
    describeEncoder('CNN')
    return onmt.CNNEncoder.new(opt, inputNetwork)
  else
    describeEncoder('unidirectional RNN')
    return onmt.Encoder.new(opt, inputNetwork)
  end

end

function Factory.buildWordEncoder(opt, dicts)
  _G.logger:info(' * Encoder:')

  local inputNetwork = buildInputNetwork(opt, dicts,
                                         opt.src_word_vec_size or opt.word_vec_size,
                                         opt.pre_word_vecs_enc, opt.fix_word_vecs_enc)

  return Factory.buildEncoder(opt, inputNetwork)
end

function Factory.loadEncoder(pretrained)
  local encoder

  if pretrained.name == 'Encoder' then
    encoder = onmt.Encoder.load(pretrained)
  elseif pretrained.name == 'BiEncoder' then
    encoder = onmt.BiEncoder.load(pretrained)
  elseif pretrained.name == 'PDBiEncoder' then
    encoder = onmt.PDBiEncoder.load(pretrained)
  elseif pretrained.name == 'DBiEncoder' then
    encoder = onmt.DBiEncoder.load(pretrained)
  elseif pretrained.name == 'GoogleEncoder' then
    encoder = onmt.GoogleEncoder.load(pretrained)
  elseif pretrained.name == 'CNNEncoder' then
    encoder = onmt.CNNEncoder.load(pretrained)
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

function Factory.buildDecoder(opt, inputNetwork, generator, attnModel)
  describeRNN(opt)

  return onmt.Decoder.new(opt, inputNetwork, generator, attnModel)
end

function Factory.buildWordDecoder(opt, dicts)
  _G.logger:info(' * Decoder:')

  local inputNetwork = buildInputNetwork(opt, dicts,
                                         opt.tgt_word_vec_size or opt.word_vec_size,
                                         opt.pre_word_vecs_dec, opt.fix_word_vecs_dec)

  local generator = Factory.buildGenerator(opt, dicts)
  local attnModel = Factory.buildAttention(opt)

  return Factory.buildDecoder(opt, inputNetwork, generator, attnModel)
end

function Factory.loadDecoder(pretrained)
  return onmt.Decoder.load(pretrained)
end

function Factory.buildGenerator(opt, dicts)
  local sizes = Factory.getOutputSizes(dicts)
  return onmt.Generator(opt, sizes)
end

function Factory.loadGenerator(pretrained)
  return onmt.Generator.load(pretrained)
end

function Factory.buildAttention(args)
  if args.attention == 'none' then
    _G.logger:info('   - attention: none')
    return onmt.NoAttention(args, args.rnn_size)
  else
    local multi_head = ''
    if args.multi_head_attention and args.multi_head_attention > 1 then
      multi_head = ', multi head = '..args.multi_head_attention
    end
    _G.logger:info('   - attention: global (%s), dropout = %0.2f%s', args.global_attention, args.dropout_attention, multi_head)
    return onmt.GlobalAttention(args, args.rnn_size)
  end
end

function Factory.loadSentenceNLLCriterion(pretrained)
  return onmt.SentenceNLLCriterion.load(pretrained)
end

return Factory
