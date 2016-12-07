local function buildEncoder(opt, dicts)
  local input_network = onmt.WordEmbedding.new(#dicts.words, -- vocab size
                                               opt.word_vec_size,
                                               opt.pre_word_vecs_enc,
                                               opt.fix_word_vecs_enc)

  local input_size = opt.word_vec_size

  -- Sequences with features.
  if #dicts.features > 0 then
    local src_feat_embedding = onmt.FeaturesEmbedding.new(dicts.features, opt.feat_vec_exponent)

    input_network = nn.Sequential()
      :add(nn.ParallelTable()
             :add(input_network)
             :add(src_feat_embedding))
      :add(nn.JoinTable(2))

    input_size = input_size + src_feat_embedding.outputSize
  end

  if opt.brnn then
    -- Compute rnn hidden size depending on hidden states merge action.
    local rnn_size = opt.rnn_size
    if opt.brnn_merge == 'concat' then
      if opt.rnn_size % 2 ~= 0 then
        error('in concat mode, rnn_size must be divisible by 2')
      end
      rnn_size = rnn_size / 2
    elseif opt.brnn_merge == 'sum' then
      rnn_size = rnn_size
    else
      error('invalid merge action ' .. opt.brnn_merge)
    end

    local rnn = onmt.LSTM.new(opt.num_layers, input_size, rnn_size, opt.dropout)

    return onmt.BiEncoder.new(input_network, rnn, opt.brnn_merge)
  else
    local rnn = onmt.LSTM.new(opt.num_layers, input_size, opt.rnn_size, opt.dropout)

    return onmt.Encoder.new(input_network, rnn)
  end
end

local function buildDecoder(opt, dicts)
  local input_network = onmt.WordEmbedding.new(#dicts.words, -- vocab size
                                               opt.word_vec_size,
                                               opt.pre_word_vecs_dec,
                                               opt.fix_word_vecs_dec)

  local input_size = opt.word_vec_size

  local generator

  -- Sequences with features.
  if #dicts.features > 0 then
    local targ_feat_embedding = onmt.FeaturesEmbedding.new(dicts.features, opt.feat_vec_exponent)

    input_network = nn.Sequential()
      :add(nn.ParallelTable()
             :add(input_network)
             :add(targ_feat_embedding))
      :add(nn.JoinTable(2))

    input_size = input_size + targ_feat_embedding.outputSize

    generator = onmt.FeaturesGenerator.new(opt.rnn_size, #dicts.words, dicts.features)
  else
    generator = onmt.Generator.new(opt.rnn_size, #dicts.words)
  end

  if opt.input_feed == 1 then
    print("Using input feeding")
    input_size = input_size + opt.rnn_size
  end

  local rnn = onmt.LSTM.new(opt.num_layers, input_size, opt.rnn_size, opt.dropout)

  return onmt.Decoder.new(input_network, rnn, generator, opt.input_feed == 1)
end

--[[ This is useful when training from a model in parallel mode: each thread must own its model. ]]
local function clonePretrained(model)
  local clone = {}

  for k, v in pairs(model) do
    if k == 'modules' then
      clone.modules = {}
      for i = 1, #v do
        table.insert(clone.modules, utils.Tensor.deepClone(v[i]))
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

return {
  buildEncoder = buildEncoder,
  buildDecoder = buildDecoder,
  loadEncoder = loadEncoder,
  loadDecoder = loadDecoder
}
