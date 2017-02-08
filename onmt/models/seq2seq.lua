--[[ sequence to sequence attention Model. ]]
require 'onmt.models.Model'
local seq2seq, parent = torch.class('onmt.Model.seq2seq', 'onmt.Model')

local seq2seq_options = {
  {'-layers', 2,           [[Number of layers in the RNN encoder/decoder]],
                     {valid=onmt.ExtendedCmdLine.isUInt()}},
  {'-rnn_size', 500, [[Size of RNN hidden states]],
                     {valid=onmt.ExtendedCmdLine.isUInt()}},
  {'-rnn_type', 'LSTM', [[Type of RNN cell]],
                     {enum={'LSTM','GRU'}}},
  {'-word_vec_size', 0, [[Common word embedding size. If set, this overrides -src_word_vec_size and -tgt_word_vec_size.]],
                     {valid=onmt.ExtendedCmdLine.isUInt()}},
  {'-src_word_vec_size', '500', [[Comma-separated list of source embedding sizes: word[,feat1,feat2,...].]]},
  {'-tgt_word_vec_size', '500', [[Comma-separated list of target embedding sizes: word[,feat1,feat2,...].]]},
  {'-feat_merge', 'concat', [[Merge action for the features embeddings]],
                     {enum={'concat','sum'}}},
  {'-feat_vec_exponent', 0.7, [[When features embedding sizes are not set and using -feat_merge concat, their dimension
                                will be set to N^exponent where N is the number of values the feature takes.]]},
  {'-feat_vec_size', 20, [[When features embedding sizes are not set and using -feat_merge sum, this is the common embedding size of the features]],
                     {valid=onmt.ExtendedCmdLine.isUInt()}},
  {'-input_feed', 1, [[Feed the context vector at each time step as additional input (via concatenation with the word embeddings) to the decoder.]],
                     {enum={0,1}}},
  {'-residual', false, [[Add residual connections between RNN layers.]]},
  {'-brnn', false, [[Use a bidirectional encoder]]},
  {'-brnn_merge', 'sum', [[Merge action for the bidirectional hidden states]],
                     {enum={'concat','sum'}}},
  {'-pre_word_vecs_enc', '', [[If a valid path is specified, then this will load
                                     pretrained word embeddings on the encoder side.
                                     See README for specific formatting instructions.]],
                         {valid=onmt.ExtendedCmdLine.fileNullOrExists}},
  {'-pre_word_vecs_dec', '', [[If a valid path is specified, then this will load
                                       pretrained word embeddings on the decoder side.
                                       See README for specific formatting instructions.]],
                         {valid=onmt.ExtendedCmdLine.fileNullOrExists}},
  {'-fix_word_vecs_enc', false, [[Fix word embeddings on the encoder side]]},
  {'-fix_word_vecs_dec', false, [[Fix word embeddings on the decoder side]]},
  {'-dropout', 0.3, [[Dropout probability. Dropout is applied between vertical LSTM stacks.]]}
}

function seq2seq.declareOpts(cmd)
  cmd:setCmdLineOptions(seq2seq_options, "Sequence to Sequence Attention")
end

function seq2seq:__init(args, datasetOrCheckpoint, verboseOrReplica)
  parent.__init(self, args)
  onmt.utils.Table.merge(self.args, onmt.ExtendedCmdLine.getModuleOpts(args, seq2seq_options))
  if type(datasetOrCheckpoint)=='Checkpoint' then
    local checkpoint = datasetOrCheckpoint
    local replica = verboseOrReplica
    self.models.encoder = onmt.Factory.loadEncoder(checkpoint.models.encoder, replica)
    self.models.decoder = onmt.Factory.loadDecoder(checkpoint.models.decoder, replica)
  else
    local dataset = datasetOrCheckpoint
    local verbose = verboseOrReplica
    self.models.encoder = onmt.Factory.buildWordEncoder(args, dataset.dicts.src, verbose)
    self.models.decoder = onmt.Factory.buildWordDecoder(args, dataset.dicts.tgt, verbose)
  end
end

-- Returns model name.
function seq2seq.modelName()
  return "Sequence to Sequence Attention"
end

-- Returns expected dataMode.
function seq2seq.dataType()
  return "BITEXT"
end

-- batch fields for seq2seq model
function seq2seq.batchInit()
  return {
           size = 1,
           sourceLength = 0,
           targetLength = 0,
           targetNonZeros = 0
         }
end

function seq2seq.batchAggregate(batchA, batch)
  batchA.sourceLength = batchA.sourceLength + batch.sourceLength * batch.size
  batchA.targetLength = batchA.targetLength + batch.targetLength * batch.size
  batchA.targetNonZeros = batchA.targetNonZeros + batch.targetNonZeros
  return batchA
end

function seq2seq:forwardComputeLoss(batch, criterion)
  local encoderStates, context = self.models.encoder:forward(batch)
  return self.models.decoder:computeLoss(batch, encoderStates, context, criterion)
end

function seq2seq:buildCriterion(dataset)
  return onmt.Criterion.new(dataset.dicts.tgt.words:size(),
                            dataset.dicts.tgt.features)
end

function seq2seq:countTokens(batch)
  return batch.targetNonZeros
end

function seq2seq:trainNetwork(batch, criterion, doProfile, dryRun)
  if doProfile then _G.profiler:start("encoder.fwd") end
  local encStates, context = self.models.encoder:forward(batch)
  if doProfile then _G.profiler:stop("encoder.fwd") end

  if doProfile then _G.profiler:start("decoder.fwd") end
  local decOutputs = self.models.decoder:forward(batch, encStates, context)
  if dryRun then decOutputs = onmt.utils.Tensor.recursiveClone(decOutputs) end
  if doProfile then _G.profiler:stop("decoder.fwd") end

  if doProfile then _G.profiler:start("decoder.bwd") end
  local encGradStatesOut, gradContext, loss = self.models.decoder:backward(batch, decOutputs, criterion)
  if doProfile then _G.profiler:stop("decoder.bwd") end

  if doProfile then _G.profiler:start("encoder.bwd") end
  self.models.encoder:backward(batch, encGradStatesOut, gradContext)
  if doProfile then _G.profiler:stop("encoder.bwd") end
  return loss
end
