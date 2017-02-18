local Factory = torch.class('Factory')

local options = {
  {'-brnn', false, [[Use a bidirectional encoder]]},
  {'-dbrnn', false, [[Use a deep bidirectional encoder]]}
}

function Factory.declareOpts(cmd)
  cmd:setCmdLineOptions(options)
  onmt.BiEncoder.declareOpts(cmd)
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

local function buildInputNetwork(opt, dicts, wordSizes, pretrainedWords, fixWords)
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

  -- Sequence with features.
  if #dicts.features > 0 then
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
      :add(nn.JoinTable(2))
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

function Factory.buildEncoder(opt, inputNetwork)

  if opt.brnn then
    return onmt.BiEncoder.new(opt, inputNetwork)
  elseif opt.dbrnn then
    return onmt.DBiEncoder.new(opt, inputNetwork)
  else
    return onmt.SimpleEncoder.new(opt, inputNetwork)
  end

end

function Factory.buildWordEncoder(opt, dicts)
  local inputNetwork = buildInputNetwork(opt, dicts,
                                         opt.src_word_vec_size or opt.word_vec_size,
                                         opt.pre_word_vecs_enc, opt.fix_word_vecs_enc)

  return Factory.buildEncoder(opt, inputNetwork)
end

function Factory.loadEncoder(pretrained, clone)
  if clone then
    pretrained = onmt.utils.Tensor.deepClone(pretrained)
  end

  if pretrained.name == 'Encoder' then
    return onmt.Encoder.load(pretrained)
  end
  if pretrained.name == 'BiEncoder' then
    return onmt.BiEncoder.load(pretrained)
  end

  -- Keep for backward compatibility.
  local brnn = #pretrained.modules == 2
  if brnn then
    return onmt.BiEncoder.load(pretrained)
  else
    return onmt.Encoder.load(pretrained)
  end
end

function Factory.buildDecoder(opt, inputNetwork, generator, verbose)
  local inputSize = inputNetwork.inputSize

  if opt.input_feed == 1 then
    if verbose then
      _G.logger:info(' * using input feeding')
    end
    inputSize = inputSize + opt.rnn_size
  end

  local RNN = onmt.LSTM
  if opt.rnn_type == 'GRU' then
    RNN = onmt.GRU
  end
  local rnn = RNN.new(opt.layers, inputSize, opt.rnn_size, opt.dropout, opt.residual)

  return onmt.Decoder.new(inputNetwork, rnn, generator, opt.input_feed == 1)
end

function Factory.buildWordDecoder(opt, dicts, verbose)
  local inputNetwork = buildInputNetwork(opt, dicts,
                                         opt.tgt_word_vec_size or opt.word_vec_size,
                                         opt.pre_word_vecs_dec, opt.fix_word_vecs_dec)

  local generator = Factory.buildGenerator(opt.rnn_size, dicts)

  return Factory.buildDecoder(opt, inputNetwork, generator, verbose)
end

function Factory.loadDecoder(pretrained, clone)
  if clone then
    pretrained = onmt.utils.Tensor.deepClone(pretrained)
  end

  return onmt.Decoder.load(pretrained)
end

function Factory.buildGenerator(rnnSize, dicts)
  if #dicts.features > 0 then
    return onmt.FeaturesGenerator(rnnSize, Factory.getOutputSizes(dicts))
  else
    return onmt.Generator(rnnSize, dicts.words:size())
  end
end

return Factory
