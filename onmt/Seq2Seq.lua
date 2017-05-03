--[[ Sequence to sequence model with attention. ]]
local Seq2Seq, parent = torch.class('Seq2Seq', 'Model')

local options = {
  {
    '-enc_layers', 0,
    [[If > 0, number of layers of the encoder. This overrides the global `-layers` option.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt(),
      structural = 0
    }
  },
  {
    '-dec_layers', 0,
    [[If > 0, number of layers of the decoder. This overrides the global `-layers` option.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt(),
      structural = 0
    }
  },
  {
    '-word_vec_size', 0,
    [[Shared word embedding size. If set, this overrides `-src_word_vec_size` and `-tgt_word_vec_size`.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt(),
      structural = 0
    }
  },
  {
    '-src_word_vec_size', { 500 },
    [[List of source embedding sizes: `word[ feat1[ feat2[ ...] ] ]`.]],
    {
      structural = 0
    }
  },
  {
    '-tgt_word_vec_size', { 500 },
    [[List of target embedding sizes: `word[ feat1[ feat2[ ...] ] ]`.]],
    {
      structural = 0
    }
  },
  {
    '-pre_word_vecs_enc', '',
    [[Path to pretrained word embeddings on the encoder side serialized as a Torch tensor.]],
    {
      valid = onmt.utils.ExtendedCmdLine.fileNullOrExists,
      init_only = true
    }
  },
  {
    '-pre_word_vecs_dec', '',
    [[Path to pretrained word embeddings on the decoder side serialized as a Torch tensor.]],
    {
      valid = onmt.utils.ExtendedCmdLine.fileNullOrExists,
      init_only = true
    }
  },
  {
    '-fix_word_vecs_enc', false,
    [[Fix word embeddings on the encoder side.]],
    {
      structural = 1
    }
  },
  {
    '-fix_word_vecs_dec', false,
    [[Fix word embeddings on the decoder side.]],
    {
      structural = 1
    }
  },
  {
    '-feat_merge', 'concat',
    [[Merge action for the features embeddings.]],
    {
      enum = {'concat', 'sum'},
      structural = 0
    }
  },
  {
    '-feat_vec_exponent', 0.7,
    [[When features embedding sizes are not set and using `-feat_merge concat`, their dimension
      will be set to `N^feat_vec_exponent` where `N` is the number of values the feature takes.]],
    {
      structural = 0
    }
  },
  {
    '-feat_vec_size', 20,
    [[When features embedding sizes are not set and using `-feat_merge sum`,
      this is the common embedding size of the features]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt(),
      structural = 0
    }
  }
}

function Seq2Seq.declareOpts(cmd)
  cmd:setCmdLineOptions(options, Seq2Seq.modelName())
  onmt.Encoder.declareOpts(cmd)
  onmt.Bridge.declareOpts(cmd)
  onmt.Decoder.declareOpts(cmd)
  onmt.Factory.declareOpts(cmd)
end

function Seq2Seq:__init(args, dicts)
  parent.__init(self, args)
  onmt.utils.Table.merge(self.args, onmt.utils.ExtendedCmdLine.getModuleOpts(args, options))
  self.args.uneven_batches = args.uneven_batches

  if not dicts.src then
    -- the input is already a vector
    args.dimInputSize = dicts.srcInputSize
  end

  local encArgs = onmt.utils.Tensor.deepClone(args)
  encArgs.layers = encArgs.enc_layers > 0 and encArgs.enc_layers or encArgs.layers
  self.models.encoder = onmt.Factory.buildWordEncoder(encArgs, dicts.src)

  local decArgs = onmt.utils.Tensor.deepClone(args)
  decArgs.layers = decArgs.dec_layers > 0 and decArgs.dec_layers or decArgs.layers
  self.models.decoder = onmt.Factory.buildWordDecoder(decArgs, dicts.tgt)

  self.models.bridge = onmt.Bridge(args.bridge,
                                   encArgs.rnn_size,
                                   self.models.encoder.args.numEffectiveLayers,
                                   decArgs.rnn_size,
                                   self.models.decoder.args.numEffectiveLayers)

  self.criterion = onmt.ParallelClassNLLCriterion(onmt.Factory.getOutputSizes(dicts.tgt))
end

function Seq2Seq.load(args, models, dicts)
  local self = torch.factory('Seq2Seq')()

  parent.__init(self, args)
  onmt.utils.Table.merge(self.args, onmt.utils.ExtendedCmdLine.getModuleOpts(args, options))
  self.args.uneven_batches = args.uneven_batches

  self.models.encoder = onmt.Factory.loadEncoder(models.encoder)
  self.models.decoder = onmt.Factory.loadDecoder(models.decoder)
  self.models.bridge = onmt.Bridge.load(models.bridge)
  self.criterion = onmt.ParallelClassNLLCriterion(onmt.Factory.getOutputSizes(dicts.tgt))

  return self
end

-- Returns model name.
function Seq2Seq.modelName()
  return 'Sequence to Sequence with Attention'
end

-- Returns expected dataMode.
function Seq2Seq.dataType()
  return 'bitext'
end

function Seq2Seq:returnIndividualLosses(enable)
  if not self.models.decoder.returnIndividualLosses then
    _G.logger:info('Current Seq2Seq model does not support training with sample_w_ppl option')
    return false
  else
    self.models.decoder:returnIndividualLosses(enable)
  end
  return true
end

function Seq2Seq:enableProfiling()
  _G.profiler.addHook(self.models.encoder, 'encoder')
  _G.profiler.addHook(self.models.decoder, 'decoder')
  _G.profiler.addHook(self.models.decoder.modules[2], 'generator')
  _G.profiler.addHook(self.criterion, 'criterion')
end

function Seq2Seq:getOutput(batch)
  return batch.targetOutput
end

function Seq2Seq:maskPadding(batch)
  self.models.encoder:maskPadding()
  if batch and batch.uneven then
    self.models.decoder:maskPadding(self.models.encoder:contextSize(batch.sourceSize, batch.sourceLength))
  else
    self.models.decoder:maskPadding()
  end
end

function Seq2Seq:forwardComputeLoss(batch)
  if self.args.uneven_batches then
    self:maskPadding(batch)
  end

  local encoderStates, context = self.models.encoder:forward(batch)
  local decoderInitStates = self.models.bridge:forward(encoderStates)
  return self.models.decoder:computeLoss(batch, decoderInitStates, context, self.criterion)
end

function Seq2Seq:trainNetwork(batch, dryRun)
  if self.args.uneven_batches then
    self:maskPadding(batch)
  end

  local encStates, context = self.models.encoder:forward(batch)
  local decInitStates = self.models.bridge:forward(encStates)
  local decOutputs = self.models.decoder:forward(batch, decInitStates, context)

  if dryRun then
    decOutputs = onmt.utils.Tensor.recursiveClone(decOutputs)
  end

  local decGradInputStates, gradContext, loss, indvLoss = self.models.decoder:backward(batch, decOutputs, self.criterion)
  local encGradOutputStates = self.models.bridge:backward(encStates, decGradInputStates)
  self.models.encoder:backward(batch, encGradOutputStates, gradContext)

  return loss, indvLoss
end

return Seq2Seq
