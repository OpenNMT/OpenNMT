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
  cmd:option('-max_num_unks', math.huge, [[All sequences with more unks than this will be ignored during beam search]])
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

  local translator = self
  local Advancer = torch.class('Advancer', 'BeamSearchAdvancer')
  function Advancer:initBeam()
    local tokens = onmt.utils.Cuda.convert(torch.IntTensor(batch.size))
      :fill(onmt.Constants.BOS)
    local features = {}
    for j = 1, #translator.dicts.tgt.features do
      features[j] = torch.IntTensor(batch.size):fill(onmt.Constants.EOS)
    end
    local sourceSizes = batch.sourceSize:clone()

    -- Define state to be { decoder states, decoder output, context,
    -- attentions, features, sourceSizes, step }.
    local state = {encStates, nil, context, nil, features,
      sourceSizes, 1}
    return onmt.translate.Beam.new(tokens, state)
  end
  function Advancer:update(beam)
    local state = beam:state()
    local decStates, decOut, prevContext, _, features, sourceSizes, t
        = table.unpack(state, 1, 7)
    local tokens = beam:tokens()
    local token = tokens[#tokens]
    local inputs
    if #features == 0 then
      inputs = token
    elseif #features == 1 then
      inputs = { token, features[1] }
    else
      inputs = { token }
      table.insert(inputs, features)
    end
    translator.models.decoder:maskPadding(sourceSizes, batch.sourceLength)
    decOut, decStates = translator.models.decoder
      :forwardOne(inputs, decStates, prevContext, decOut)
    t = t + 1
    local softmaxOut = translator.models.decoder.softmaxAttn.output
    local nextState = {decStates, decOut, prevContext, softmaxOut, nil,
      sourceSizes, t}
    beam:setState(nextState)
  end
  function Advancer:expand(beam)
    local state = beam:state()
    local decOut = state[2]
    local out = translator.models.decoder.generator:forward(decOut)
    local features = {}
    for j = 2, #out do
      local _, best = out[j]:max(2)
      features[j - 1] = best:view(-1)
    end
    state[5] = features
    local scores = out[1]
    return scores
  end
  -- IsComplete function
  function Advancer:isComplete(beam)
    local tokens = beam:tokens()
    local seqLength = #tokens
    local complete = tokens[#tokens]:eq(onmt.Constants.EOS)
    if seqLength > translator.opt.max_sent_length then
      complete:fill(1)
    end
    return complete
  end
  -- Filter function (ignore too many UNKs)
  function Advancer:filter(beam)
    local tokens = beam:tokens()
    local numUnks = onmt.utils.Cuda.convert(torch.zeros(tokens[1]:size(1)))
    for t = 1, #tokens do
      local token = tokens[t]
      numUnks:add(onmt.utils.Cuda.convert(token:eq(onmt.Constants.UNK):double()))
    end
    -- Disallow too many UNKs
    local unSatisfied = numUnks:gt(translator.opt.max_num_unks)
    -- Disallow empty hypotheses
    if #tokens == 2 then
      unSatisfied:add(tokens[2]:eq(onmt.Constants.EOS))
    end
    return unSatisfied:ge(1)
  end

  local advancer = Advancer.new()
  local attnIndex, featsIndex = 4, 5
  if self.opt.replace_unk then
    advancer:setKeptStateIndexes({attnIndex, featsIndex})
  else
    advancer:setKeptStateIndexes({featsIndex})
  end

  local beamSearcher = onmt.translate.BeamSearcher.new(advancer)
  local results = beamSearcher:search(self.opt.beam_size, self.opt.n_best)

  debug.getregistry()['Advancer'] = nil
  collectgarbage()

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
      -- Remove unnecessary values from the attention vectors
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
  local data = self:buildData(srcBatch, srcFeaturesBatch, goldBatch, goldFeaturesBatch)
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

  return predBatch, infoBatch
end

return Translator
