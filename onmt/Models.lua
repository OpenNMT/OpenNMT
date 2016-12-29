
local function declareOpts(cmd)
  cmd:text("")
  cmd:text("**Model options**")
  cmd:text("")
  
  cmd:option('-layers', 2, [[Number of layers in the LSTM encoder/decoder]])
  cmd:option('-rnn_size', 500, [[Size of LSTM hidden states]])
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
  cmd:option('-pre_word_vecs_dec', '', [[If a valid path is specified, then this will load
                                     pretrained word embeddings on the decoder side.
                                     See README for specific formatting instructions.]])
  cmd:option('-fix_word_vecs_enc', false, [[Fix word embeddings on the encoder side]])
  cmd:option('-fix_word_vecs_dec', false, [[Fix word embeddings on the decoder side]])
end

local function buildInputNetwork(opt, dicts, pretrainedWords, fixWords)
  local wordEmbedding = onmt.WordEmbedding.new(dicts.words:size(), -- vocab size
                                               opt.word_vec_size,
                                               pretrainedWords,
                                               fixWords)

  local inputs
  local inputSize = opt.word_vec_size

  local multiInputs = #dicts.features > 0

  if multiInputs then
    inputs = nn.ParallelTable()
      :add(wordEmbedding)
  else
    inputs = wordEmbedding
  end

  -- Sequence with features.
  if #dicts.features > 0 then
    local featEmbedding = onmt.FeaturesEmbedding.new(dicts.features,
                                                     opt.feat_vec_exponent,
                                                     opt.feat_vec_size,
                                                     opt.feat_merge)
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
  local inputNetwork, inputSize = buildInputNetwork(opt, dicts, opt.pre_word_vecs_enc, opt.fix_word_vecs_enc)

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

    local rnn = onmt.LSTM.new(opt.layers, inputSize, rnnSize, opt.dropout, opt.residual)

    return onmt.BiEncoder.new(inputNetwork, rnn, opt.brnn_merge)
  else
    local rnn = onmt.LSTM.new(opt.layers, inputSize, opt.rnn_size, opt.dropout, opt.residual)

    return onmt.Encoder.new(inputNetwork, rnn)
  end
end

local function buildDecoder(opt, dicts, verbose)
  local inputNetwork, inputSize = buildInputNetwork(opt, dicts, opt.pre_word_vecs_dec, opt.fix_word_vecs_dec)

  local generator

  if #dicts.features > 0 then
    generator = onmt.FeaturesGenerator.new(opt.rnn_size, dicts.words:size(), dicts.features)
  else
    generator = onmt.Generator.new(opt.rnn_size, dicts.words:size())
  end

  if opt.input_feed == 1 then
    if verbose then
      print(" * using input feeding")
    end
    inputSize = inputSize + opt.rnn_size
  end

  local rnn = onmt.LSTM.new(opt.layers, inputSize, opt.rnn_size, opt.dropout, opt.residual)

  return onmt.Decoder.new(inputNetwork, rnn, generator, opt.input_feed == 1)
end

--[[ This is useful when training from a model in parallel mode: each thread must own its model. ]]
local function clonePretrained(model)
  local clone = {}

  for k, v in pairs(model) do
    if k == 'modules' then
      clone.modules = {}
      for i = 1, #v do
        table.insert(clone.modules, onmt.utils.Tensor.deepClone(v[i]))
      end
    else
      clone[k] = v
    end
  end

  return clone
end

local function loadEncoder(pretrained, clone)
  local brnn = #pretrained.modules == 2

  if clone then
    pretrained = clonePretrained(pretrained)
  end

  if brnn then
    return onmt.BiEncoder.load(pretrained)
  else
    return onmt.Encoder.load(pretrained)
  end
end

local function loadDecoder(pretrained, clone)
  if clone then
    pretrained = clonePretrained(pretrained)
  end

  return onmt.Decoder.load(pretrained)
end

local function buildCriterion(vocabSize, features)
  local criterion = nn.ParallelCriterion(false)

  local function addNllCriterion(size)
    -- Ignores padding value.
    local w = torch.ones(size)
    w[onmt.Constants.PAD] = 0

    local nll = nn.ClassNLLCriterion(w)

    -- Let the training code manage loss normalization.
    nll.sizeAverage = false
    criterion:add(nll)
  end

  addNllCriterion(vocabSize)

  for j = 1, #features do
    addNllCriterion(features[j]:size())
  end

  return criterion
end

return {
  declareOpts = declareOpts,
  buildEncoder = buildEncoder,
  buildDecoder = buildDecoder,
  buildCriterion = buildCriterion,
  loadEncoder = loadEncoder,
  loadDecoder = loadDecoder
}
