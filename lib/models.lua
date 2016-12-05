local function buildEncoder(opt, dicts, pretrained)
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
      error('invalid merge action ' .. merge)
    end

    local rnn = onmt.LSTM.new(opt.num_layers, input_size, rnn_size, opt.dropout)

    return onmt.BiEncoder.new(input_network,
                              rnn,
                              opt.brnn_merge,
                              pretrained.encoder,
                              pretrained.encoder_bwd)
  else
    local rnn = onmt.LSTM.new(opt.num_layers, input_size, opt.rnn_size, opt.dropout)

    return onmt.Encoder.new(input_network, rnn, pretrained.encoder)
  end
end

local function buildDecoder(opt, dicts, pretrained)
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

    generator = onmt.FeatureGenerator.new(opt.rnn_size, #dicts.words, dicts.features)
  else
    generator = onmt.Generator.new(opt.rnn_size, #dicts.words)
  end

  if opt.input_feed == 1 then
    print("Using input feeding")
    input_size = input_size + opt.rnn_size
  end

  local rnn = onmt.LSTM.new(opt.num_layers, input_size, opt.rnn_size, opt.dropout)

  return onmt.Decoder.new(input_network,
                          rnn,
                          pretrained.generator or generator,
                          opt.input_feed == 1,
                          pretrained.decoder)
end

return {
  buildEncoder = buildEncoder,
  buildDecoder = buildDecoder
}
