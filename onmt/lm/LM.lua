local LM = torch.class('LM')

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

function LM.declareOpts(cmd)
  cmd:setCmdLineOptions(options, 'LM')
end

function LM:__init(args)
  self.opt = args

  _G.logger:info('Loading \'' .. self.opt.model .. '\'...')
  self.checkpoint = torch.load(self.opt.model)

  self.dataType = self.checkpoint.options.data_type or 'bitext'
  if not self.checkpoint.options.model_type or self.checkpoint.options.model_type ~= 'lm' then
    _G.logger:error('LM can only process lm models')
    os.exit(0)
  end

  self.model = onmt.LanguageModel.load(self.checkpoint.options, self.checkpoint.models, self.checkpoint.dicts)
  onmt.utils.Cuda.convert(self.model)

  self.dicts = self.checkpoint.dicts
end

function LM:srcFeat()
  return self.dataType == 'monotext'
end

function LM:buildInput(tokens)
  local data = {}

  local words, features = onmt.utils.Features.extract(tokens)

  data.words = words

  if #features > 0 then
    data.features = features
  end

  return data
end

function LM:buildOutput(data)
  return table.concat(onmt.utils.Features.annotate(data.words, data.features), ' ')
end

function LM:buildData(src, ignoreEmpty, onlyPrefix)
  local srcData = {}
  srcData.words = {}
  srcData.features = {}

  local ignored = {}
  local indexMap = {}
  local index = 1

  for b = 1, #src do
    if ignoreEmpty and src[b].words and #src[b].words == 0 then
      table.insert(ignored, b)
    else
      indexMap[index] = b
      index = index + 1

      table.insert(srcData.words,
                 self.dicts.src.words:convertToIdx(src[b].words, onmt.Constants.UNK_WORD, onmt.Constants.BOS_WORD, not onlyPrefix))
      if #self.dicts.src.features > 0 then
        table.insert(srcData.features,
                     onmt.utils.Features.generateSource(self.dicts.src.features, src[b].features))
      end
    end
  end

  return onmt.data.Dataset.new(srcData), ignored, indexMap
end

--[[ Evaluate

Parameters:

  * `src` - a batch of tables containing:
    - `words`: the table of source words
    - `features`: the table of features sequences (`src.features[i][j]` is the value of the ith feature of the jth token)

Returns:

  * `results` - a batch of ppl
]]
function LM:evaluate(src)
  local data, ignored = self:buildData(src, true)

  local results = {}

  if data:batchCount() > 0 then
    local batch = onmt.utils.Cuda.convert(data:getBatch())

    local indvPpl = select(2, self.model:forwardComputeLoss(batch, true))

    for i = 1, batch.size do
      table.insert(results, indvPpl[i])
    end
  end

  for i = 1, #ignored do
    table.insert(results, ignored[i], {})
  end

  return results
end

--[[ Sample

Parameters:

  * `src` - a batch of tables containing:
    - `words`: table of source token sequences - possibly empty
    - `features`: table of features sequences (`src.features[i][j]` is the value of the ith feature of the jth token)

Returns:

  * `results` - a batch of source tokens
]]
function LM:sample(src, max_length, temperature)
  local data = self:buildData(src, false, true)

  local states, context = self.model.models.encoder:forward(data:getBatch())

  local foundEOS = 0
  local results = {}

  for i = 1, #data.src do
    table.insert(results, torch.totable(data.src[i]))
  end

  local completed = {}
  local t = 0

  while foundEOS ~= #data.src and t < max_length do
    local genOutputs = self.model.models.generator:forward(context:select(2, context:size(2)))
    genOutputs[1]:div(temperature) -- scale by temperature
    local probs = torch.exp(genOutputs[1])
    probs:div(torch.sum(probs)) -- renormalize so probs sum to one
    local words = torch.multinomial(probs:float(), 1)
    local nextsrc = {}
    for i = 1, #data.src do
      if not results[i] then
        results[i] = {}
      end
      if not completed[i] then
        if words[i][1] == onmt.Constants.EOS_WORD then
          if not completed[i] then foundEOS = foundEOS +1 end
          completed[i] = 1
        end
        table.insert(results[i], words[i][1])
      end
      table.insert(nextsrc, words[i])
    end
    local batch = onmt.data.Batch.new(nextsrc)
    states, context = self.model.models.encoder:forward(batch, states)
    t = t + 1
  end

  for i = 1, #data.src do
    results[i] = table.concat(self.dicts.src.words:convertToLabels(results[i], onmt.Constants.EOS), ' ')
  end

  return results
end

return LM
