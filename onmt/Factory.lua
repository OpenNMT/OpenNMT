local Factory = torch.class('Factory')

local options = {
  {'-brnn', false, [[Use a bidirectional encoder.]]},
  {'-dbrnn', false, [[Use a deep bidirectional encoder.]]},
  {'-pdbrnn', false, [[Use pyramidal deep bidirectional encoder.]]}
}

function Factory.declareOpts(cmd)
  cmd:setCmdLineOptions(options)
  onmt.BiEncoder.declareOpts(cmd)
  onmt.DBiEncoder.declareOpts(cmd)
  onmt.PDBiEncoder.declareOpts(cmd)
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
    _G.logger:info('   - with word embeddings size: ' .. wordEmbSize)
  end

  -- Sequence with features.
  if #dicts.features > 0 then
    if verbose then
      _G.logger:info('   - with features embeddings sizes: ' .. table.concat(featEmbSizes, ', '))
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

local function fixWordEmbeddings(model, fix)
  model:apply(function(mod)
    if torch.typename(mod) == 'onmt.WordEmbedding' then
      mod:fixEmbeddings(fix)
    end
  end)
end

function Factory.getOutputSizes(dicts)
  local outputSizes = { dicts.words:size() }
  for i = 1, #dicts.features do
    table.insert(outputSizes, dicts.features[i]:size())
  end
  return outputSizes
end

function Factory.buildEncoder(opt, inputNetwork)

  if opt.brnn then
    _G.logger:info('   - Bidirectional %s Encoder: %d layers, rnn_size %d, dropout %0.1f',
                   opt.rnn_type, opt.layers, opt.rnn_size, opt.dropout)
    return onmt.BiEncoder.new(opt, inputNetwork)
  elseif opt.dbrnn then
    _G.logger:info('   - Deep Bidirectional %s Encoder: %d layers, rnn_size %d, dropout %0.1f',
                   opt.rnn_type, opt.layers, opt.rnn_size, opt.dropout)
    return onmt.DBiEncoder.new(opt, inputNetwork)
  elseif opt.pdbrnn then
    _G.logger:info('   - Pyramidal Bidirectional %s Encoder: %d layers, rnn_size %d, dropout %0.1f',
                   opt.rnn_type, opt.layers, opt.rnn_size, opt.dropout)
    return onmt.PDBiEncoder.new(opt, inputNetwork)
  else
    _G.logger:info('   - Simple %s Encoder: %d layers, rnn_size %d, dropout %0.1f',
                   opt.rnn_type, opt.layers, opt.rnn_size, opt.dropout)
    return onmt.Encoder.new(opt, inputNetwork)
  end

end

function Factory.buildWordEncoder(opt, dicts, verbose)
  if verbose then
    _G.logger:info(' * Encoder:')
  end
  local inputNetwork
  if dicts then
    inputNetwork = buildInputNetwork(opt, dicts,
                                     opt.src_word_vec_size or opt.word_vec_size,
                                     opt.pre_word_vecs_enc, opt.fix_word_vecs_enc,
                                     verbose)
  else
    inputNetwork = nn.Identity()
    inputNetwork.inputSize = opt.dimInputSize
  end

  return Factory.buildEncoder(opt, inputNetwork)
end

function Factory.loadEncoder(pretrained, clone, opt)
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

  if opt then
    fixWordEmbeddings(encoder, opt.fix_word_vecs_enc)
  end

  return encoder
end

function Factory.buildDecoder(opt, inputNetwork, generator)
  local inputSize = inputNetwork.inputSize

  if opt.input_feed == 1 then
    inputSize = inputSize + opt.rnn_size
  end

  local RNN = onmt.LSTM
  if opt.rnn_type == 'GRU' then
    RNN = onmt.GRU
  end
  local rnn = RNN.new(opt.layers, inputSize, opt.rnn_size, opt.dropout, opt.residual, opt.dropout_input)

  return onmt.Decoder.new(inputNetwork, rnn, generator, opt.input_feed == 1)
end

function Factory.buildWordDecoder(opt, dicts, verbose)
  if verbose then
    _G.logger:info(' * Decoder:')
  end

  local inputNetwork = buildInputNetwork(opt, dicts,
                                         opt.tgt_word_vec_size or opt.word_vec_size,
                                         opt.pre_word_vecs_dec, opt.fix_word_vecs_dec,
                                         verbose)

  local generator = Factory.buildGenerator(opt.rnn_size, dicts)

  return Factory.buildDecoder(opt, inputNetwork, generator)
end

function Factory.loadDecoder(pretrained, clone, opt)
  if clone then
    pretrained = onmt.utils.Tensor.deepClone(pretrained)
  end

  local decoder = onmt.Decoder.load(pretrained)

  if opt then
    fixWordEmbeddings(decoder, opt.fix_word_vecs_dec)
  end

  return decoder
end

function Factory.buildGenerator(rnnSize, dicts)
  if #dicts.features > 0 then
    return onmt.FeaturesGenerator(rnnSize, Factory.getOutputSizes(dicts))
  else
    return onmt.Generator(rnnSize, dicts.words:size())
  end
end

return Factory
