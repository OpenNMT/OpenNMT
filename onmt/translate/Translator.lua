local Translator = torch.class('Translator')

function Translator.declareOpts(cmd)
  cmd:option('-model', '', [[Path to model .t7 file]])

  -- beam search options
  cmd:text("")
  cmd:text("**Beam Search options**")
  cmd:text("")
  cmd:option('-beam_size', 5,[[Beam size]])
  cmd:option('-batch_size', 30, [[Batch size]])
  cmd:option('-max_sent_length', 250, [[Maximum sentence length. If any sequences in srcfile are longer than this then it will error out]])
  cmd:option('-replace_unk', false, [[Replace the generated UNK tokens with the source token that
                              had the highest attention weight. If phrase_table is provided,
                              it will lookup the identified source token and give the corresponding
                              target token. If it is not provided (or the identified source token
                              does not exist in the table) then it will copy the source token]])
  cmd:option('-phrase_table', '', [[Path to source-target dictionary to replace UNK
                                     tokens. See README.md for the format this file should be in]])
  cmd:option('-n_best', 1, [[If > 1, it will also output an n_best list of decoded sentences]])
end


function Translator:__init(args)
  self.opt = args
  onmt.utils.Cuda.init(self.opt)

  local log
  if _G.logger then
    log = function (...) return _G.logger:info(...) end
  else
    log = print
  end
  log('Loading \'' .. self.opt.model .. '\'...')
  self.checkpoint = torch.load(self.opt.model)

  self.models = {}
  self.models.encoder = onmt.Models.loadEncoder(self.checkpoint.models.encoder)
  self.models.decoder = onmt.Models.loadDecoder(self.checkpoint.models.decoder)

  self.models.encoder:evaluate()
  self.models.decoder:evaluate()

  onmt.utils.Cuda.convert(self.models.encoder)
  onmt.utils.Cuda.convert(self.models.decoder)

  self.dicts = self.checkpoint.dicts

  if self.opt.phrase_table:len() > 0 then
    self.phraseTable = onmt.translate.PhraseTable.new(self.opt.phrase_table)
  end
end

function Translator:buildData(srcBatch, srcFeaturesBatch, goldBatch, goldFeaturesBatch)
  local srcData = {}
  srcData.words = {}
  srcData.features = {}

  local tgtData
  if goldBatch ~= nil then
    tgtData = {}
    tgtData.words = {}
    tgtData.features = {}
  end

  for b = 1, #srcBatch do
    table.insert(srcData.words,
                 self.dicts.src.words:convertToIdx(srcBatch[b], onmt.Constants.UNK_WORD))

    if #self.dicts.src.features > 0 then
      table.insert(srcData.features,
                   onmt.utils.Features.generateSource(self.dicts.src.features, srcFeaturesBatch[b]))
    end

    if tgtData ~= nil then
      table.insert(tgtData.words,
                   self.dicts.tgt.words:convertToIdx(goldBatch[b],
                                                     onmt.Constants.UNK_WORD,
                                                     onmt.Constants.BOS_WORD,
                                                     onmt.Constants.EOS_WORD))

      if #self.dicts.tgt.features > 0 then
        table.insert(tgtData.features,
                     onmt.utils.Features.generateTarget(self.dicts.tgt.features, goldFeaturesBatch[b]))
      end
    end
  end

  return onmt.data.Dataset.new(srcData, tgtData)
end

function Translator:buildTargetTokens(pred, predFeats, src, attn)
  local tokens = self.dicts.tgt.words:convertToLabels(pred, onmt.Constants.EOS)

  -- Always ignore last token to stay consistent, even it may not be EOS.
  table.remove(tokens)

  if self.opt.replace_unk then
    for i = 1, #tokens do
      if tokens[i] == onmt.Constants.UNK_WORD then
        local _, maxIndex = attn[i]:max(1)
        local source = src[maxIndex[1]]

        if self.phraseTable and self.phraseTable:contains(source) then
          tokens[i] = self.phraseTable:lookup(source)
        else
          tokens[i] = source
        end
      end
    end
  end

  if predFeats ~= nil then
    tokens = onmt.utils.Features.annotate(tokens, predFeats, self.dicts.tgt.features)
  end

  return tokens
end

function Translator:translateBatch(batch)
  self.models.encoder:maskPadding()
  self.models.decoder:maskPadding()

  local encStates, context = self.models.encoder:forward(batch)

  local goldScore
  if batch.targetInput ~= nil then
    if batch.size > 1 then
      self.models.decoder:maskPadding(batch.sourceSize, batch.sourceLength)
    end
    goldScore = self.models.decoder:computeScore(batch, encStates, context)
  end

  -- Expand tensors for each beam.
  context = context
    :contiguous()
    :view(1, batch.size, batch.sourceLength, self.checkpoint.options.rnn_size)
    :expand(self.opt.beam_size, batch.size, batch.sourceLength, self.checkpoint.options.rnn_size)
    :contiguous()
    :view(self.opt.beam_size * batch.size, batch.sourceLength, self.checkpoint.options.rnn_size)

  for j = 1, #encStates do
    encStates[j] = encStates[j]
      :view(1, batch.size, self.checkpoint.options.rnn_size)
      :expand(self.opt.beam_size, batch.size, self.checkpoint.options.rnn_size)
      :contiguous()
      :view(self.opt.beam_size * batch.size, self.checkpoint.options.rnn_size)
  end

  local remainingSents = batch.size

  -- As finished sentences are removed from the batch, this table maps the batches
  -- to their index within the remaining sentences.
  local batchIdx = {}

  local beam = {}

  for b = 1, batch.size do
    table.insert(beam, onmt.translate.Beam.new(self.opt.beam_size, #self.dicts.tgt.features))
    table.insert(batchIdx, b)
  end

  local i = 1

  local decOut
  local decStates = encStates

  while remainingSents > 0 and i < self.opt.max_sent_length do
    i = i + 1

    -- Prepare decoder input.
    local input = torch.IntTensor(self.opt.beam_size, remainingSents)
    local inputFeatures = {}
    local sourceSizes = torch.IntTensor(remainingSents)

    for b = 1, batch.size do
      if not beam[b].done then
        local idx = batchIdx[b]
        sourceSizes[idx] = batch.sourceSize[b]

        -- Get current state of the beam search.
        local wordState, featuresState = beam[b]:getCurrentState()
        input[{{}, idx}]:copy(wordState)

        for j = 1, #self.dicts.tgt.features do
          if inputFeatures[j] == nil then
            inputFeatures[j] = torch.IntTensor(self.opt.beam_size, remainingSents)
          end
          inputFeatures[j][{{}, idx}]:copy(featuresState[j])
        end
      end
    end

    input = input:view(self.opt.beam_size * remainingSents)
    for j = 1, #self.dicts.tgt.features do
      inputFeatures[j] = inputFeatures[j]:view(self.opt.beam_size * remainingSents)
    end

    local inputs
    if #inputFeatures == 0 then
      inputs = input
    elseif #inputFeatures == 1 then
      inputs = { input, inputFeatures[1] }
    else
      inputs = { input }
      table.insert(inputs, inputFeatures)
    end

    if batch.size > 1 then
      self.models.decoder:maskPadding(sourceSizes, batch.sourceLength, self.opt.beam_size)
    end

    decOut, decStates = self.models.decoder:forwardOne(inputs, decStates, context, decOut)

    local out = self.models.decoder.generator:forward(decOut)

    for j = 1, #out do
      out[j] = out[j]:view(self.opt.beam_size, remainingSents, out[j]:size(2)):transpose(1, 2):contiguous()
    end
    local wordLk = out[1]

    local softmaxOut = self.models.decoder.softmaxAttn.output:view(self.opt.beam_size, remainingSents, -1)
    local newRemainingSents = remainingSents

    for b = 1, batch.size do
      if not beam[b].done then
        local idx = batchIdx[b]

        local featsLk = {}
        for j = 1, #self.dicts.tgt.features do
          table.insert(featsLk, out[j + 1][idx])
        end

        if beam[b]:advance(wordLk[idx], featsLk, softmaxOut[{{}, idx}]) then
          newRemainingSents = newRemainingSents - 1
          batchIdx[b] = 0
        end

        for j = 1, #decStates do
          local view = decStates[j]
            :view(self.opt.beam_size, remainingSents, self.checkpoint.options.rnn_size)
          view[{{}, idx}] = view[{{}, idx}]:index(1, beam[b]:getCurrentOrigin())
        end
      end
    end

    if newRemainingSents > 0 and newRemainingSents ~= remainingSents then
      -- Update sentence indices within the batch and mark sentences to keep.
      local toKeep = {}
      local newIdx = 1
      for b = 1, #batchIdx do
        local idx = batchIdx[b]
        if idx > 0 then
          table.insert(toKeep, idx)
          batchIdx[b] = newIdx
          newIdx = newIdx + 1
        end
      end

      toKeep = torch.LongTensor(toKeep)

      -- Update rnn states and context.
      for j = 1, #decStates do
        decStates[j] = decStates[j]
          :view(self.opt.beam_size, remainingSents, self.checkpoint.options.rnn_size)
          :index(2, toKeep)
          :view(self.opt.beam_size*newRemainingSents, self.checkpoint.options.rnn_size)
      end

      decOut = decOut
        :view(self.opt.beam_size, remainingSents, self.checkpoint.options.rnn_size)
        :index(2, toKeep)
        :view(self.opt.beam_size*newRemainingSents, self.checkpoint.options.rnn_size)

      context = context
        :view(self.opt.beam_size, remainingSents, batch.sourceLength, self.checkpoint.options.rnn_size)
        :index(2, toKeep)
        :view(self.opt.beam_size*newRemainingSents, batch.sourceLength, self.checkpoint.options.rnn_size)

      -- The `index()` method allocates a new storage so clean the previous ones to
      -- keep a stable memory usage.
      collectgarbage()
    end

    remainingSents = newRemainingSents
  end

  local allHyp = {}
  local allFeats = {}
  local allAttn = {}
  local allScores = {}

  for b = 1, batch.size do
    local scores, ks = beam[b]:sortBest()

    local hypBatch = {}
    local featsBatch = {}
    local attnBatch = {}
    local scoresBatch = {}

    for n = 1, self.opt.n_best do
      local hyp, feats, attn = beam[b]:getHyp(ks[n])

      -- remove unnecessary values from the attention vectors
      for j = 1, #attn do
        local size = batch.sourceSize[b]
        attn[j] = attn[j]:narrow(1, batch.sourceLength - size + 1, size)
      end

      table.insert(hypBatch, hyp)
      if #feats > 0 then
        table.insert(featsBatch, feats)
      end
      table.insert(attnBatch, attn)
      table.insert(scoresBatch, scores[n])
    end

    table.insert(allHyp, hypBatch)
    table.insert(allFeats, featsBatch)
    table.insert(allAttn, attnBatch)
    table.insert(allScores, scoresBatch)
  end

  return allHyp, allFeats, allScores, allAttn, goldScore
end

function Translator:translate(srcBatch, srcFeaturesBatch, goldBatch, goldFeaturesBatch)
  local data = self:buildData(srcBatch, srcFeaturesBatch, goldBatch, goldFeaturesBatch)
  local batch = data:getBatch()

  local pred, predFeats, predScore, attn, goldScore = self:translateBatch(batch)

  local predBatch = {}
  local infoBatch = {}

  for b = 1, batch.size do
    table.insert(predBatch, self:buildTargetTokens(pred[b][1], predFeats[b][1], srcBatch[b], attn[b][1]))

    local info = {}
    info.score = predScore[b][1]
    info.nBest = {}

    if goldScore ~= nil then
      info.goldScore = goldScore[b]
    end

    if self.opt.n_best > 1 then
      for n = 1, self.opt.n_best do
        info.nBest[n] = {}
        info.nBest[n].tokens = self:buildTargetTokens(pred[b][n], predFeats[b][n], srcBatch[b], attn[b][n])
        info.nBest[n].score = predScore[b][n]
      end
    end

    table.insert(infoBatch, info)
  end

  return predBatch, infoBatch
end

return Translator
