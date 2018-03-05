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
  },
  {
    '-max_length', 100,
    [[Maximal length of sentences in sample mode.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt
    }
  },
  {
    '-temperature', 1,
    [[For `sample` mode, higher temperatures cause the model to take more chances and increase diversity of results, but at a cost of more mistakes.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isFloat(0.0001, 1)
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
  self.model:evaluate()
  onmt.utils.Cuda.convert(self.model)

  self.dicts = self.checkpoint.dicts
end

function LM:srcFeat()
  return self.dataType == 'monotext'
end

function LM:buildInput(tokens)
  local data = {}

  local words, features = onmt.utils.Features.extract(tokens)
  local vocabs = onmt.utils.Placeholders.norm(words)

  data.words = vocabs
  data.features = features

  return data
end

function LM:buildOutput(data)
  return table.concat(onmt.utils.Features.annotate(data.words, data.features), ' ')
end

function LM:buildData(src, ignoreEmpty, onlyPrefix)
  if onlyPrefix then
    onlyPrefix = nil
  else
    onlyPrefix = '<eos>'
  end

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
                 self.dicts.src.words:convertToIdx(src[b].words, onmt.Constants.UNK_WORD, onmt.Constants.BOS_WORD, onlyPrefix))

      if #self.dicts.src.features > 0 then
        if #src[b].words == 0 and #src[b].features == 0 then
          for _ = 1, #self.dicts.src.features do
            table.insert(src[b].features, {})
          end
        end
        local features = onmt.utils.Features.generateTarget(self.dicts.src.features, src[b].features, false, false)
        if not onlyPrefix then
          for i = 1, #features do
            features[i] = features[i]:narrow(1,1,features[i]:size(1)-1)
          end
        end
        table.insert(srcData.features, features)
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
  local batch = data:getBatch()
  local numFeatures = #batch.sourceInputFeatures

  local states, context = self.model.models.encoder:forward(batch)

  local foundEOS = 0
  local results = {}
  local resultFeats = {}

  for i = 1, #data.src do
    table.insert(results, torch.totable(data.src[i]))
    table.insert(resultFeats, {})
    for k = 1, numFeatures do
      table.insert(resultFeats[i], torch.totable(data.srcFeatures[i][k]))
    end
  end

  local completed = {}
  local t = 0

  while foundEOS ~= #data.src and t < max_length do
    local nextsrc = {}
    -- prepare empty features
    local nextsrcFeats = {}
    for _ = 1, numFeatures do
      table.insert(nextsrcFeats, {})
    end

    local genOutputs = self.model.models.generator:forward(context:select(2, context:size(2)))

    local justCompleted = {}
    for k = 1, #genOutputs do
      genOutputs[k]:div(temperature) -- scale by temperature
      local probs = torch.exp(genOutputs[k])
      probs:div(torch.sum(probs)) -- renormalize so probs sum to one
      local tokens = torch.multinomial(probs:float(), 1)

      for i = 1, #data.src do
        local token = tokens[i][1]
        if k == 1 then
          local changeComplete = false
          if not completed[i] then
            if token == onmt.Constants.EOS then
              if not completed[i] then foundEOS = foundEOS +1 end
              changeComplete = true
              completed[i] = 1
            end
            table.insert(results[i], token)
          end
          table.insert(nextsrc, token)
          table.insert(justCompleted, changeComplete)
        else
          if not completed[i] or justCompleted[i] then
            if justCompleted[i] then
              token = onmt.Constants.EOS
            end
            table.insert(resultFeats[i][k-1], token)
          end
          table.insert(nextsrcFeats[k-1], token)
        end
      end
    end

    nextsrc = torch.LongTensor(nextsrc)
    for i, v in ipairs(nextsrcFeats) do
      nextsrcFeats[i] = torch.LongTensor(v)
    end

    local inputs
    if #nextsrcFeats == 0 then
      inputs = nextsrc
    elseif #nextsrcFeats == 1 then
      inputs = { nextsrc, nextsrcFeats[1] }
    else
      inputs = { nextsrc }
      table.insert(inputs, nextsrcFeats)
    end

    onmt.utils.Cuda.convert(inputs)

    states, context = self.model.models.encoder:forwardOne(inputs, states)
    context = nn.utils.addSingletonDimension(context, 2)
    t = t + 1
  end

  for i = 1, #data.src do
    results[i] = self.dicts.src.words:convertToLabels(results[i], onmt.Constants.EOS)
    -- remove BOS/EOS
    table.remove(results[i], 1)
    table.remove(results[i])
    for k = 1, numFeatures do
      resultFeats[i][k] = self.dicts.src.features[k]:convertToLabels(resultFeats[i][k])
      table.remove(resultFeats[i][k], 1)
      table.remove(resultFeats[i][k])
    end
    results[i] = self:buildOutput({words=results[i],features=resultFeats[i]})
  end

  return results
end

return LM
