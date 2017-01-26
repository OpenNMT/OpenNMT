local Translator = torch.class('Translator')

function Translator.declareOpts(cmd)
  cmd:option('-model', '', [[Path to model .t7 file]])

  -- beam search options
  cmd:text("")
  cmd:text("**Beam Search options**")
  cmd:text("")
  cmd:option('-beam_size', 5,[[Beam size]])
  cmd:option('-batch_size', 30, [[Batch size]])
  cmd:option('-max_sent_length', 250, [[Maximum output sentence length.]])
  cmd:option('-replace_unk', false, [[Replace the generated UNK tokens with the source token that
                              had the highest attention weight. If phrase_table is provided,
                              it will lookup the identified source token and give the corresponding
                              target token. If it is not provided (or the identified source token
                              does not exist in the table) then it will copy the source token]])
  cmd:option('-phrase_table', '', [[Path to source-target dictionary to replace UNK
                                     tokens. See README.md for the format this file should be in]])
  cmd:option('-n_best', 1, [[If > 1, it will also output an n_best list of decoded sentences]])
  cmd:option('-max_num_unks', math.huge, [[All sequences with more unks than this will be ignored
                                           during beam search]])
  cmd:option('-before_filter_factor', 1, [[Optional, set this only if filter is being used. Before
                              applying filters, hypotheses with top `beamSize * preFilterFactor`
                              scores will be considered. If the returned hypotheses voilate filters,
                              then set this to a larger value to consider more.]])
end


function Translator:__init(args)
  self.opt = args
  onmt.utils.Cuda.init(self.opt)

  _G.logger:info('Loading \'' .. self.opt.model .. '\'...')
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

  local ignored = {}

  for b = 1, #srcBatch do
    if #srcBatch[b] == 0 then
      table.insert(ignored, b)
    else
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
  end

  return onmt.data.Dataset.new(srcData, tgtData), ignored
end

function Translator:buildTargetTokens(pred, predFeats, src, attn)
  local tokens = self.dicts.tgt.words:convertToLabels(pred, onmt.Constants.EOS)

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

  -- Compute gold score.
  local goldScore
  if batch.targetInput ~= nil then
    if batch.size > 1 then
      self.models.decoder:maskPadding(batch.sourceSize, batch.sourceLength)
    end
    goldScore = self.models.decoder:computeScore(batch, encStates, context)
  end

  -- Specify how to go one step forward.
  local advancer = onmt.translate.DecoderAdvancer.new(self.models.decoder,
                                   batch, context, self.opt.max_sent_length,
                                   self.opt.max_num_unks, encStates, self.dicts)
  local attnIndex, featsIndex = 4, 5
  if self.opt.replace_unk then
    advancer:setKeptStateIndexes({attnIndex, featsIndex})
  else
    advancer:setKeptStateIndexes({featsIndex})
  end

  -- Conduct beam search.
  local beamSearcher = onmt.translate.BeamSearcher.new(advancer)
  local results = beamSearcher:search(self.opt.beam_size, self.opt.n_best,
                                      self.opt.before_filter_factor)

  local allHyp = {}
  local allFeats = {}
  local allAttn = {}
  local allScores = {}

  for b = 1, batch.size do
    local hypBatch = {}
    local featsBatch = {}
    local attnBatch = {}
    local scoresBatch = {}

    for n = 1, self.opt.n_best do
      local tokens = results[b][n].tokens
      local score = results[b][n].score
      local states = results[b][n].states
      local attn = states[attnIndex] or {}
      local feats = states[featsIndex] or {}
      table.remove(tokens)

      -- Remove unnecessary values from the attention vectors.
      local size = batch.sourceSize[b]
      for j = 1, #attn do
        attn[j] = attn[j]:narrow(1, batch.sourceLength - size + 1, size)
      end

      table.insert(hypBatch, tokens)
      if #feats > 0 then
        table.insert(featsBatch, feats)
      end
      table.insert(attnBatch, attn)
      table.insert(scoresBatch, score)
    end

    table.insert(allHyp, hypBatch)
    table.insert(allFeats, featsBatch)
    table.insert(allAttn, attnBatch)
    table.insert(allScores, scoresBatch)
  end

  return allHyp, allFeats, allScores, allAttn, goldScore
end

function Translator:translate(srcBatch, srcFeaturesBatch, goldBatch, goldFeaturesBatch)
  local data, ignored = self:buildData(srcBatch, srcFeaturesBatch, goldBatch, goldFeaturesBatch)
  local batch = data:getBatch()

  local pred, predFeats, predScore, attn, goldScore = self:translateBatch(batch)

  local predBatch = {}
  local infoBatch = {}

  for b = 1, batch.size do
    table.insert(predBatch, self:buildTargetTokens(pred[b][1], predFeats[b][1], srcBatch[b], attn[b][1]))

    local info = {}
    info.score = predScore[b][1]
    info.attention = attn[b][1]
    info.nBest = {}

    if goldScore ~= nil then
      info.goldScore = goldScore[b]
    end

    if self.opt.n_best > 1 then
      for n = 1, self.opt.n_best do
        info.nBest[n] = {}
        info.nBest[n].tokens = self:buildTargetTokens(pred[b][n], predFeats[b][n], srcBatch[b], attn[b][n])
        info.nBest[n].attention = attn[b][n]
        info.nBest[n].score = predScore[b][n]
      end
    end

    table.insert(infoBatch, info)
  end

  for i = 1, #ignored do
    table.insert(predBatch, ignored[i], {})
    table.insert(infoBatch, ignored[i], {})
  end

  return predBatch, infoBatch
end

return Translator
