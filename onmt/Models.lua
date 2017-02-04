-- Return effective embeddings size based on user options.
local function resolveEmbSizes(opt, dicts, wordSizes)
  local wordEmbSize
  local featEmbSizes = {}

  wordSizes = onmt.utils.String.split(tostring(wordSizes), ',')

  if opt.word_vec_size > 0 then
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

  return inputNetwork, inputSize
end

local function buildEncoder(opt, dicts)
  local inputNetwork, inputSize = buildInputNetwork(opt, dicts, opt.src_word_vec_size or opt.word_vec_size,
                                                    opt.pre_word_vecs_enc, opt.fix_word_vecs_enc)

  local RNN = onmt.LSTM
  if opt.rnn_type == 'GRU' then
    RNN = onmt.GRU
  end

  if opt.brnn then
    -- Compute rnn hidden size depending on hidden states merge action.
    local rnnSize = opt.rnn_size
    if opt.brnn_merge == 'concat' then
      if opt.rnn_size % 2 ~= 0 then
        error('in concat mode, rnn_size must be divisible by 2')
      end
      rnnSize = rnnSize / 2
    elseif opt.brnn_merge == 'sum' then
      rnnSize = rnnSize
    else
      error('invalid merge action ' .. opt.brnn_merge)
    end

    local rnn = RNN.new(opt.layers, inputSize, rnnSize, opt.dropout, opt.residual)

    return onmt.BiEncoder.new(inputNetwork, rnn, opt.brnn_merge)
  else
    local rnn = RNN.new(opt.layers, inputSize, opt.rnn_size, opt.dropout, opt.residual)

    return onmt.Encoder.new(inputNetwork, rnn)
  end
end

local function buildDecoder(opt, dicts, verbose)
  local inputNetwork, inputSize = buildInputNetwork(opt, dicts, opt.tgt_word_vec_size,
                                                    opt.pre_word_vecs_dec, opt.fix_word_vecs_dec)

  local RNN = onmt.LSTM
  if opt.rnn_type == 'GRU' then
    RNN = onmt.GRU
  end

  local generator

  if #dicts.features > 0 then
    generator = onmt.FeaturesGenerator.new(opt.rnn_size, dicts.words:size(), dicts.features)
  else
    generator = onmt.Generator.new(opt.rnn_size, dicts.words:size())
  end

  if opt.input_feed == 1 then
    if verbose then
      _G.logger:info(" * using input feeding")
    end
    inputSize = inputSize + opt.rnn_size
  end

  local rnn = RNN.new(opt.layers, inputSize, opt.rnn_size, opt.dropout, opt.residual)

  return onmt.Decoder.new(inputNetwork, rnn, generator, opt.input_feed == 1)
end

local function loadEncoder(pretrained, clone)
  local brnn = #pretrained.modules == 2

  if clone then
    pretrained = onmt.utils.Tensor.deepClone(pretrained)
  end

  if brnn then
    return onmt.BiEncoder.load(pretrained)
  else
    return onmt.Encoder.load(pretrained)
  end
end

local function loadDecoder(pretrained, clone)
  if clone then
    pretrained = onmt.utils.Tensor.deepClone(pretrained)
  end

  return onmt.Decoder.load(pretrained)
end

return {
  buildEncoder = buildEncoder,
  buildDecoder = buildDecoder,
  loadEncoder = loadEncoder,
  loadDecoder = loadDecoder
}
