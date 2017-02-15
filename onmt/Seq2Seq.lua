--[[ Sequence to sequence model with attention. ]]
local Seq2Seq, parent = torch.class('Seq2Seq', 'Model')

local Seq2Seq_options = {
  {'-layers', 2,           [[Number of layers in the RNN encoder/decoder]],
                     {valid=onmt.utils.ExtendedCmdLine.isUInt()}},
  {'-rnn_size', 500, [[Size of RNN hidden states]],
                     {valid=onmt.utils.ExtendedCmdLine.isUInt()}},
  {'-rnn_type', 'LSTM', [[Type of RNN cell]],
                     {enum={'LSTM','GRU'}}},
  {'-word_vec_size', 0, [[Common word embedding size. If set, this overrides -src_word_vec_size and -tgt_word_vec_size.]],
                     {valid=onmt.utils.ExtendedCmdLine.isUInt()}},
  {'-src_word_vec_size', '500', [[Comma-separated list of source embedding sizes: word[,feat1,feat2,...].]]},
  {'-tgt_word_vec_size', '500', [[Comma-separated list of target embedding sizes: word[,feat1,feat2,...].]]},
  {'-feat_merge', 'concat', [[Merge action for the features embeddings]],
                     {enum={'concat','sum'}}},
  {'-feat_vec_exponent', 0.7, [[When features embedding sizes are not set and using -feat_merge concat, their dimension
                                will be set to N^exponent where N is the number of values the feature takes.]]},
  {'-feat_vec_size', 20, [[When features embedding sizes are not set and using -feat_merge sum, this is the common embedding size of the features]],
                     {valid=onmt.utils.ExtendedCmdLine.isUInt()}},
  {'-input_feed', 1, [[Feed the context vector at each time step as additional input (via concatenation with the word embeddings) to the decoder.]],
                     {enum={0,1}}},
  {'-residual', false, [[Add residual connections between RNN layers.]]},
  {'-brnn', false, [[Use a bidirectional encoder]]},
  {'-brnn_merge', 'sum', [[Merge action for the bidirectional hidden states]],
                     {enum={'concat','sum'}}},
  {'-pre_word_vecs_enc', '', [[If a valid path is specified, then this will load
                                     pretrained word embeddings on the encoder side.
                                     See README for specific formatting instructions.]],
                         {valid=onmt.utils.ExtendedCmdLine.fileNullOrExists}},
  {'-pre_word_vecs_dec', '', [[If a valid path is specified, then this will load
                                       pretrained word embeddings on the decoder side.
                                       See README for specific formatting instructions.]],
                         {valid=onmt.utils.ExtendedCmdLine.fileNullOrExists}},
  {'-fix_word_vecs_enc', false, [[Fix word embeddings on the encoder side]]},
  {'-fix_word_vecs_dec', false, [[Fix word embeddings on the decoder side]]},
  {'-dropout', 0.3, [[Dropout probability. Dropout is applied between vertical LSTM stacks.]]}
}

function Seq2Seq.declareOpts(cmd)
  cmd:setCmdLineOptions(Seq2Seq_options, "Sequence to Sequence Attention")
end

function Seq2Seq:__init(args, dicts, verbose)
  parent.__init(self, args)
  onmt.utils.Table.merge(self.args, onmt.utils.ExtendedCmdLine.getModuleOpts(args, Seq2Seq_options))

  self.models.encoder = onmt.Factory.buildWordEncoder(args, dicts.src, verbose)
  self.models.decoder = onmt.Factory.buildWordDecoder(args, dicts.tgt, verbose)
end

function Seq2Seq.load(args, models, isReplica)
  local self = torch.factory('Seq2Seq')()

  parent.__init(self, args)
  onmt.utils.Table.merge(self.args, onmt.utils.ExtendedCmdLine.getModuleOpts(args, Seq2Seq_options))

  self.models.encoder = onmt.Factory.loadEncoder(models.encoder, isReplica)
  self.models.decoder = onmt.Factory.loadDecoder(models.decoder, isReplica)

  return self
end

-- Returns model name.
function Seq2Seq.modelName()
  return "Sequence to Sequence Attention"
end

-- Returns expected dataMode.
function Seq2Seq.dataType()
  return "bitext"
end

-- batch fields for Seq2Seq model
function Seq2Seq.batchInit()
  return {
           size = 1,
           sourceLength = 0,
           targetLength = 0
         }
end

function Seq2Seq.batchAggregate(batchA, batch)
  batchA.sourceLength = batchA.sourceLength + batch.sourceLength * batch.size
  batchA.targetLength = batchA.targetLength + batch.targetLength * batch.size
  return batchA
end

function Seq2Seq:getOutput(batch)
  return batch.targetOutput
end

function Seq2Seq:forwardComputeLoss(batch, criterion)
  local encoderStates, context = self.models.encoder:forward(batch)
  return self.models.decoder:computeLoss(batch, encoderStates, context, criterion)
end

function Seq2Seq:buildCriterion(dicts)
  local outputSizes = { dicts.tgt.words:size() }
  for j = 1, #dicts.tgt.features do
    table.insert(outputSizes, dicts.tgt.features[j]:size())
  end

  return onmt.ParallelClassNLLCriterion(outputSizes)
end

function Seq2Seq:trainNetwork(batch, criterion, dryRun)
  local encStates, context = self.models.encoder:forward(batch)

  local decOutputs = self.models.decoder:forward(batch, encStates, context)

  if dryRun then
    decOutputs = onmt.utils.Tensor.recursiveClone(decOutputs)
  end

  local encGradStatesOut, gradContext, loss = self.models.decoder:backward(batch, decOutputs, criterion)
  self.models.encoder:backward(batch, encGradStatesOut, gradContext)

  return loss
end

return Seq2Seq
