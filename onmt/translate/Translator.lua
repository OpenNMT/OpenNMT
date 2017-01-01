local checkpoint = nil
local models = {}
local dicts = {}
local opt = {}

local phraseTable

local function declareOpts(cmd)
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
  cmd:option('-max_num_unks', math.huge, [[All sequences with more unks than this will be ignored during beam search]])
end


local function init(args)
  opt = args
  onmt.utils.Cuda.init(opt)

  print('Loading \'' .. opt.model .. '\'...')
  checkpoint = torch.load(opt.model)

  models.encoder = onmt.Models.loadEncoder(checkpoint.models.encoder)
  models.decoder = onmt.Models.loadDecoder(checkpoint.models.decoder)

  models.encoder:evaluate()
  models.decoder:evaluate()

  onmt.utils.Cuda.convert(models.encoder)
  onmt.utils.Cuda.convert(models.decoder)

  dicts = checkpoint.dicts

  if opt.phrase_table:len() > 0 then
    phraseTable = onmt.translate.PhraseTable.new(opt.phrase_table)
  end
end

local function buildData(srcBatch, srcFeaturesBatch, goldBatch, goldFeaturesBatch)
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
    table.insert(srcData.words, dicts.src.words:convertToIdx(srcBatch[b], onmt.Constants.UNK_WORD))

    if #dicts.src.features > 0 then
      table.insert(srcData.features,
                   onmt.utils.Features.generateSource(dicts.src.features, srcFeaturesBatch[b]))
    end

    if tgtData ~= nil then
      table.insert(tgtData.words,
                   dicts.tgt.words:convertToIdx(goldBatch[b],
                                                  onmt.Constants.UNK_WORD,
                                                  onmt.Constants.BOS_WORD,
                                                  onmt.Constants.EOS_WORD))

      if #dicts.tgt.features > 0 then
        table.insert(tgtData.features,
                     onmt.utils.Features.generateTarget(dicts.tgt.features, goldFeaturesBatch[b]))
      end
    end
  end

  return onmt.data.Dataset.new(srcData, tgtData)
end

local function buildTargetTokens(pred, predFeats, src, attn)
  local tokens = dicts.tgt.words:convertToLabels(pred, onmt.Constants.EOS)

  -- Always ignore last token to stay consistent, even it may not be EOS.
  --table.remove(tokens)

  if opt.replace_unk then
    for i = 1, #tokens do
      if tokens[i] == onmt.Constants.UNK_WORD then
        local _, maxIndex = attn[i]:max(1)
        local source = src[maxIndex[1]]

        if phraseTable and phraseTable:contains(source) then
          tokens[i] = phraseTable:lookup(source)
        else
          tokens[i] = source
        end
      end
    end
  end

  if predFeats ~= nil then
    tokens = onmt.utils.Features.annotate(tokens, predFeats, dicts.tgt.features)
  end

  return tokens
end

local function translateBatch(batch)
  models.encoder:maskPadding()
  models.decoder:maskPadding()

  local encStates, context = models.encoder:forward(batch)

  local goldScore
  if batch.targetInput ~= nil then
    if batch.size > 1 then
      models.decoder:maskPadding(batch.sourceSize, batch.sourceLength)
    end
    goldScore = models.decoder:computeScore(batch, encStates, context)
  end

  --function beamSearch(stepFunction, feedFunction, beamSize, endSymbol, maxSeqLength, maxScore)
  local function feedFunction(stepOutputs, topIndexes)
    if stepOutputs == nil then -- initial inputs, t == 1
      local input = onmt.utils.Cuda.convert(torch.IntTensor(batch.size)):fill(onmt.Constants.BOS)
      local numUnks = onmt.utils.Cuda.convert(torch.zeros(batch.size))
      local inputFeatures = {}
      for j = 1, #dicts.tgt.features do
        inputFeatures[j] = torch.IntTensor(batch.size):fill(onmt.Constants.EOS)
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
      local decStates = encStates
      local decOut = nil
      local sourceSizes = batch.sourceSize:clone()

      local stepInputs = {inputs, decStates, context, decOut, sourceSizes, numUnks}
      return stepInputs
    else -- inputs for t > 1
      local input = topIndexes
      local scores, decStates, decOut, context, softmaxOut, features, sourceSizes, numUnks = table.unpack(stepOutputs)
      numUnks:add(onmt.utils.Cuda.convert(topIndexes:eq(onmt.Constants.UNK):double()))
      local inputs
      if #features == 0 then
        inputs = input
      elseif #features == 1 then
        inputs = { input, features[1] }
      else
        inputs = { input }
        table.insert(inputs, features)
      end
      local stepInputs = {inputs, decStates, context, decOut, sourceSizes, numUnks}
      return stepInputs
    end
  end
  local function stepFunction(stepInputs)
    local inputs, decStates, context, decOut, sourceSizes, numUnks = table.unpack(stepInputs)
    models.decoder:maskPadding(sourceSizes, batch.sourceLength)
    decOut, decStates = models.decoder:forwardOne(inputs, decStates, context, decOut)
    local out = models.decoder.generator:forward(decOut)
    local scores = out[1]
    scores:select(2, onmt.Constants.UNK):maskedFill(numUnks:ge(opt.max_num_unks), -math.huge)
    local features = {}
    for j = 2, #out do
      local _, best = out[j]:max(2)
      features[j - 1] = best
    end
    local stepOutputs = {scores, decStates, decOut, context, models.decoder.softmaxAttn.output, features, sourceSizes, numUnks}
    return stepOutputs
  end
  local beamSearcher = onmt.translate.BeamSearcher.new(stepFunction, feedFunction, opt.max_sent_length)

  local predictions = beamSearcher:search(opt.beam_size, opt.n_best)

  collectgarbage()

  local allHyp = {}
  local allFeats = {}
  local allAttn = {}
  local allScores = {}

  local results = {}
  for n = 1, opt.n_best do
    results[n] = table.pack(beamSearcher:getPredictions(n))
  end
  for b = 1, batch.size do
    local hypBatch = {}
    local featsBatch = {}
    local attnBatch = {}
    local scoresBatch = {}

    for n = 1, opt.n_best do
      local result = results[n]
      local hyp = result[1][b]
      local scores = result[2][b]
      local attn = result[3][b][5] or {}
      local feats = result[3][b][6] or {}
      -- feats offset 1
      local featsTemp = {}
      for k = 1, #feats do
        featsTemp[k + 1] = feats[k]
      end
      feats = featsTemp

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
      table.insert(scoresBatch, scores)
    end

    table.insert(allHyp, hypBatch)
    table.insert(allFeats, featsBatch)
    table.insert(allAttn, attnBatch)
    table.insert(allScores, scoresBatch)
  end

  return allHyp, allFeats, allScores, allAttn, goldScore
end

local function translate(srcBatch, srcFeaturesBatch, goldBatch, goldFeaturesBatch)
  local data = buildData(srcBatch, srcFeaturesBatch, goldBatch, goldFeaturesBatch)
  local batch = data:getBatch()

  local pred, predFeats, predScore, attn, goldScore = translateBatch(batch)

  local predBatch = {}
  local infoBatch = {}

  for b = 1, batch.size do
    table.insert(predBatch, buildTargetTokens(pred[b][1], predFeats[b][1], srcBatch[b], attn[b][1]))

    local info = {}
    info.score = predScore[b][1]
    info.nBest = {}

    if goldScore ~= nil then
      info.goldScore = goldScore[b]
    end

    if opt.n_best > 1 then
      for n = 1, opt.n_best do
        info.nBest[n] = {}
        info.nBest[n].tokens = buildTargetTokens(pred[b][n], predFeats[b][n], srcBatch[b], attn[b][n])
        info.nBest[n].score = predScore[b][n]
      end
    end

    table.insert(infoBatch, info)
  end

  return predBatch, infoBatch
end

return {
  init = init,
  translate = translate,
  declareOpts = declareOpts
}
