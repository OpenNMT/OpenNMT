local Tagger = torch.class('Tagger')

local options = {
  {
    '-model', '',
    [[Path to the serialized model file.]],
    {
      valid = onmt.utils.ExtendedCmdLine.fileExists
    }
  },
  {
    '-batch_size', 30,
    [[Batch size.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isInt(1)
    }
  }
}

function Tagger.declareOpts(cmd)
  cmd:setCmdLineOptions(options, 'Tagger')
end

function Tagger:__init(args)
  self.opt = args

  _G.logger:info('Loading \'' .. self.opt.model .. '\'...')
  self.checkpoint = torch.load(self.opt.model)

  self.dataType = self.checkpoint.options.data_type or 'bitext'
  if not self.checkpoint.options.model_type or self.checkpoint.options.model_type ~= 'seqtagger' then
    _G.logger:error('Tagger can only process seqtagger models')
    os.exit(0)
  end

  self.model = onmt.SeqTagger.load(self.checkpoint.options, self.checkpoint.models, self.checkpoint.dicts)
  onmt.utils.Cuda.convert(self.model)

  self.dicts = self.checkpoint.dicts
end

function Tagger:srcFeat()
  return self.dataType == 'feattext'
end

function Tagger:buildInput(tokens)
  local data = {}
  if self.dataType == 'feattext' then
    data.vectors = torch.Tensor(tokens)
  else
    local words, features = onmt.utils.Features.extract(tokens)

    data.words = words

    if #features > 0 then
      data.features = features
    end
  end

  return data
end

function Tagger:buildOutput(data)
  return table.concat(onmt.utils.Features.annotate(data.words, data.features), ' ')
end

function Tagger:buildData(src)
  local srcData = {}
  srcData.words = {}
  srcData.features = {}

  local ignored = {}
  local indexMap = {}
  local index = 1

  for b = 1, #src do
    if src[b].words and #src[b].words == 0 then
      table.insert(ignored, b)
    else
      indexMap[index] = b
      index = index + 1

      if self.dicts.src then
        table.insert(srcData.words,
                   self.dicts.src.words:convertToIdx(src[b].words, onmt.Constants.UNK_WORD))
        if #self.dicts.src.features > 0 then
          table.insert(srcData.features,
                       onmt.utils.Features.generateSource(self.dicts.src.features, src[b].features))
        end
      else
        table.insert(srcData.words,onmt.utils.Cuda.convert(src[b].vectors))
      end
    end
  end

  return onmt.data.Dataset.new(srcData), ignored, indexMap
end

function Tagger:buildTargetWords(pred)
  local tokens = self.dicts.tgt.words:convertToLabels(pred, onmt.Constants.EOS)

  return tokens
end

function Tagger:buildTargetFeatures(predFeats)
  local numFeatures = #predFeats[1]

  if numFeatures == 0 then
    return {}
  end

  local feats = {}
  for _ = 1, numFeatures do
    table.insert(feats, {})
  end

  for i = 1, #predFeats do
    for j = 1, numFeatures do
      table.insert(feats[j], self.dicts.tgt.features[j]:lookup(predFeats[i][j]))
    end
  end

  return feats
end

function Tagger:tagBatch(batch)
  self.model.models.encoder:maskPadding()

  local pred = {}
  local feats = {}
  for _ = 1, batch.size do
    table.insert(pred, {})
    table.insert(feats, {})
  end
  local _, context = self.model.models.encoder:forward(batch)

  for t = 1, batch.sourceLength do
    local out = self.model.models.generator:forward(context:select(2, t))
    if type(out[1]) == 'table' then
      out = out[1]
    end
    local _, best = out[1]:max(2)
    for b = 1, batch.size do
      if t > batch.sourceLength - batch.sourceSize[b] then
        pred[b][t - batch.sourceLength + batch.sourceSize[b]] = best[b][1]
        feats[b][t - batch.sourceLength + batch.sourceSize[b]] = {}
      end
    end
    for j = 2, #out do
      _, best = out[j]:max(2)
      for b = 1, batch.size do
        if t > batch.sourceLength - batch.sourceSize[b] then
          feats[b][t - batch.sourceLength + batch.sourceSize[b]][j - 1] = best[b][1]
        end
      end
    end
  end

  return pred, feats
end

--[[ Tag a batch of source sequences.

Parameters:

  * `src` - a batch of tables containing:
    - `words`: the table of source words
    - `features`: the table of feaures sequences (`src.features[i][j]` is the value of the ith feature of the jth token)

Returns:

  * `results` - a batch of tables containing:
      - `words`: the table of target words
      - `features`: the table of target features sequences
]]
function Tagger:tag(src)
  local data, ignored = self:buildData(src)

  local results = {}

  if data:batchCount() > 0 then
    local batch = data:getBatch()

    local pred, predFeats = self:tagBatch(batch)

    for b = 1, batch.size do
      results[b] = {}
      results[b].words = self:buildTargetWords(pred[b])
      results[b].features = self:buildTargetFeatures(predFeats[b])
    end
  end

  for i = 1, #ignored do
    table.insert(results, ignored[i], {})
  end

  return results
end

return Tagger
