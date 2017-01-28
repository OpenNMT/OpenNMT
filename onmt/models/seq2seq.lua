--[[ seq2seq Model. ]]
local seq2seq, parent = torch.class('onmt.Models.seq2seq', 'onmt.Model')

function seq2seq:__init(opt, datasetOrCheckpoint, verboseOrReplica)
  parent.__init(self)
  if type(datasetOrCheckpoint)=='Checkpoint' then
    local checkpoint = datasetOrCheckpoint
    local replica = verbose
    self.models.encoder = onmt.Models.loadEncoder(checkpoint.models, replica)
    self.models.decoder = onmt.Models.loadDecoder(checkpoint.models, replica)
  else
    local dataset = datasetOrCheckpoint
    local verbose = verboseOrReplica
    self.models.encoder = onmt.Models.buildEncoder(opt, dataset.dicts.src, verbose)
    self.models.decoder = onmt.Models.buildDecoder(opt, dataset.dicts.tgt, verbose, opt.adaptive_softmax)
  end
end

function seq2seq:forwardComputeLoss(batch, criterion)
  local encoderStates, context = self.models.encoder:forward(batch)
  return self.models.decoder:computeLoss(batch, encoderStates, context, criterion)
end

function seq2seq:buildCriterion(dataset)
  return onmt.Criterion.new(dataset.dicts.tgt.words:size(),
                            dataset.dicts.tgt.features,
                            self.models.decoder.adaptive_softmax_cutoff)
end

function seq2seq:trainNetwork(batch, gradParams, criterion, doProfile, dryRun)
  if doProfile then _G.profiler:start("encoder.fwd") end
  local encStates, context = self.models.encoder:forward(batch)
  if doProfile then _G.profiler:stop("encoder.fwd"):start("decoder.fwd") end
  local decOutputs = self.models.decoder:forward(batch, encStates, context)
  if dryRun then decOutputs = onmt.utils.Tensor.recursiveClone(decOutputs) end
  if doProfile then _G.profiler:stop("decoder.fwd"):start("decoder.bwd") end
  local encGradStatesOut, gradContext, loss = self.models.decoder:backward(batch, decOutputs, criterion)
  if doProfile then _G.profiler:stop("decoder.bwd"):start("encoder.bwd") end
  self.models.encoder:backward(batch, encGradStatesOut, gradContext)
  if doProfile then _G.profiler:stop("encoder.bwd") end
  return loss
end

function seq2seq.declareOpts(cmd)
  cmd:option('-layers', 2, [[Number of layers in the RNN encoder/decoder]])
  cmd:option('-rnn_size', 500, [[Size of RNN hidden states]])
  cmd:option('-rnn_type', 'LSTM', [[Type of RNN cell: LSTM, GRU]])
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
