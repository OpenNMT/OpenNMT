local Translator = torch.class('Translator')

local options = {
  {
    '-model', { '' },
    [[List of trained models.]]
  },
  {
    '-beam_size', 5,
    [[Beam size.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isInt(1)
    }
  },
  {
    '-max_sent_length', 250,
    [[Maximum output sentence length.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isInt(1)
    }
  },
  {
    '-replace_unk', false,
    [[Replace the generated <unk> tokens with the source token that
      has the highest attention weight. If `-phrase_table` is provided,
      it will lookup the identified source token and give the corresponding
      target token. If it is not provided (or the identified source token
      does not exist in the table) then it will copy the source token]]},
  {
    '-phrase_table', '',
    [[Path to source-target dictionary to replace `<unk>` tokens.]],
    {
      valid=onmt.utils.ExtendedCmdLine.fileNullOrExists
    }
  },
  {
    '-n_best', 1,
    [[If > 1, it will also output an n-best list of decoded sentences.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isInt(1)
    }
  },
  {
    '-max_num_unks', math.huge,
    [[All sequences with more `<unk>`s than this will be ignored during beam search.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isInt(1)
    }
  },
  {
    '-target_subdict', '',
    [[Path to target words dictionary corresponding to the source.]],
    {
      valid=onmt.utils.ExtendedCmdLine.fileNullOrExists
    }
  },
  {
    '-pre_filter_factor', 1,
    [[Optional, set this only if filter is being used. Before
      applying filters, hypotheses with top `beam_size * pre_filter_factor`
      scores will be considered. If the returned hypotheses voilate filters,
      then set this to a larger value to consider more.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isInt(1)
    }
  },
  {
    '-length_norm', 0.0,
    [[Length normalization coefficient (alpha). If set to 0, no length normalization.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isFloat(0),
    }
  },
  {
    '-coverage_norm', 0.0,
    [[Coverage normalization coefficient (beta).
      An extra coverage term multiplied by beta is added to hypotheses scores.
      If is set to 0, no coverage normalization.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isFloat(0),
    }
  },
  {
    '-eos_norm', 0.0,
    [[End of sentence normalization coefficient (gamma). If set to 0, no EOS normalization.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isFloat(0),
    }
  },
  {
    '-dump_input_encoding', false,
    [[Instead of generating target tokens conditional on
    the source tokens, we print the representation
    (encoding/embedding) of the input.]]
  },
  {
    '-save_beam_to', '',
    [[Path to a file where the beam search exploration will be saved in a JSON format.
      Requires the `dkjson` package.]]
  }
}

function Translator.declareOpts(cmd)
  cmd:setCmdLineOptions(options, 'Translator')
end


function Translator:__init(args)
  self.args = args

  onmt.utils.ThreadPool.init(
    #self.args.model,
    function()
      require('onmt.init')
    end,
    function(id)
      onmt.utils.Cuda.init(args, id)
    end
  )

  local sourceVocabSize
  local targetVocabSize

  -- Load models into their own thread.
  onmt.utils.ThreadPool.dispatch(
    function(i)
      local ckpt = torch.load(self.args.model[i])
      _G.model = onmt.Seq2Seq.load(args, ckpt.models, ckpt.dicts)
      _G.model:evaluate()
      onmt.utils.Cuda.convert(_G.model.models)
      return i, ckpt, _G.model, self.args.model[i]
    end,
    function(i, ckpt, model, modelFile)
      -- Check vocabulary.
      if not sourceVocabSize then
        sourceVocabSize = ckpt.dicts.src.words:size()
        targetVocabSize = ckpt.dicts.tgt.words:size()
      else
        assert(ckpt.dicts.src.words:size() == sourceVocabSize, 'source vocabulary must be the same')
        assert(ckpt.dicts.tgt.words:size() == targetVocabSize, 'target vocabulary must be the same')
      end

      -- Reference the first model.
      if i == 1 then
        self.dataType = ckpt.options.data_type or 'bitext'
        self.modelType = ckpt.options.model_type or 'seq2seq'
        self.dicts = ckpt.dicts
        self.model = model
      end

      assert(ckpt.options.model_type == 'seq2seq', "Translator can only manage seq2seq models")

      _G.logger:info('Loaded \'%s\' on %s',
                     modelFile,
                     onmt.utils.Cuda.activated and 'GPU ' .. onmt.utils.Cuda.getGpu(i) or 'CPU')
    end
  )

  if self.args.phrase_table:len() > 0 then
    self.phraseTable = onmt.translate.PhraseTable.new(self.args.phrase_table)
  end

  if self.args.target_subdict:len() > 0 then
    self.dicts.subdict = onmt.utils.SubDict.new(self.dicts.tgt.words, self.args.target_subdict)

    onmt.utils.ThreadPool.dispatch(
      function()
        _G.model:setTargetVoc(self.dicts.subdict.targetVocTensor)
      end
    )

    _G.logger:info('Using target vocabulary from %s (%d vocabs)', self.args.target_subdict, self.dicts.subdict.targetVocTensor:size(1))
  end

  if self.args.save_beam_to:len() > 0 then
    self.beamHistories = {}
  end
end

function Translator:srcFeat()
  return self.dataType == 'feattext'
end

function Translator:buildInput(tokens)
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

function Translator:buildInputGold(tokens)
  local data = {}

  local words, features = onmt.utils.Features.extract(tokens)

  data.words = words

  if #features > 0 then
    data.features = features
  end

  return data
end

function Translator:buildOutput(data)
  return table.concat(onmt.utils.Features.annotate(data.words, data.features), ' ')
end

function Translator:buildData(src, gold)
  local srcData = {}
  srcData.words = {}
  srcData.features = {}

  local goldData
  if gold then
    goldData = {}
    goldData.words = {}
    goldData.features = {}
  end

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

      if gold then
        table.insert(goldData.words,
                     self.dicts.tgt.words:convertToIdx(gold[b].words,
                                                       onmt.Constants.UNK_WORD,
                                                       onmt.Constants.BOS_WORD,
                                                       onmt.Constants.EOS_WORD))

        if #self.dicts.tgt.features > 0 then
          table.insert(goldData.features,
                       onmt.utils.Features.generateTarget(self.dicts.tgt.features, gold[b].features))
        end
      end
    end
  end

  return onmt.data.Dataset.new(srcData, goldData), ignored, indexMap
end

function Translator:buildTargetWords(pred, src, attn)
  local tokens = self.dicts.tgt.words:convertToLabels(pred, onmt.Constants.EOS)

  if self.args.replace_unk then
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

  return tokens
end

function Translator:buildTargetFeatures(predFeats)
  local numFeatures = #predFeats[1]

  if numFeatures == 0 then
    return {}
  end

  local feats = {}
  for _ = 1, numFeatures do
    table.insert(feats, {})
  end

  for i = 2, #predFeats do
    for j = 1, numFeatures do
      table.insert(feats[j], self.dicts.tgt.features[j]:lookup(predFeats[i][j]))
    end
  end

  return feats
end

function Translator:translateBatch(batch)
  local allEncStates = {}
  local allDecStates = {}
  local allContexts = {}

  -- Dispatch source input encoding.
  onmt.utils.ThreadPool.dispatch(
    function(i)
      local batchClone = onmt.utils.Tensor.deepClone(batch)
      local model = _G.model

      local encStates, context = model.models.encoder:forward(batchClone)
      local decStates = model.models.bridge:forward(encStates)

      return i, encStates, decStates, context
    end,
    function(i, encStates, decStates, context)
      allEncStates[i] = encStates
      allDecStates[i] = decStates
      allContexts[i] = context
    end
  )

  -- Return average input encoding if defined.
  if self.args.dump_input_encoding then
    for i = 2, #allEncStates do
      allEncStates[1][#allEncStates[1]]
        :mul(i - 1)
        :add(allEncStates[i][#allEncStates[i]]:clone())
        :div(i)
    end
    return allEncStates[1][#allEncStates[1]]
  end

  -- Compute average gold score.
  local goldScore
  if batch.targetInput ~= nil then
    local processed = 0
    onmt.utils.ThreadPool.dispatch(
      function(i)
        local batchClone = onmt.utils.Tensor.deepClone(batch)
        local score = _G.model.models.decoder:computeScore(batchClone, allDecStates[i], allContexts[i])
        return score
      end,
      function(score)
        goldScore = goldScore and (goldScore * processed + score) / (processed + 1) or score
        processed = processed + 1
      end
    )
  end

  -- Specify how to go one step forward.
  local advancer = onmt.translate.DecoderAdvancer.new(nil,
                                                      batch,
                                                      allContexts,
                                                      self.args.max_sent_length,
                                                      self.args.max_num_unks,
                                                      allDecStates,
                                                      self.dicts,
                                                      self.args.length_norm,
                                                      self.args.coverage_norm,
                                                      self.args.eos_norm)

  -- Save memory by only keeping track of necessary elements in the states.
  -- Attentions are at index 4 in the states defined in onmt.translate.DecoderAdvancer.
  local attnIndex = 4

  -- Features are at index 5 in the states defined in onmt.translate.DecoderAdvancer.
  local featsIndex = 5

  advancer:setKeptStateIndexes({attnIndex, featsIndex})

  -- Conduct beam search.
  local beamSearcher = onmt.translate.BeamSearcher.new(advancer, self.args.save_beam_to:len() > 0)
  local results, histories = beamSearcher:search(self.args.beam_size,
                                                 self.args.n_best,
                                                 self.args.pre_filter_factor)

  local allHyp = {}
  local allFeats = {}
  local allAttn = {}
  local allScores = {}
  local allHistories = {}

  for b = 1, batch.size do
    local hypBatch = {}
    local featsBatch = {}
    local attnBatch = {}
    local scoresBatch = {}

    for n = 1, self.args.n_best do
      local result = results[b][n]
      local tokens = result.tokens
      local score = result.score
      local states = result.states
      local attn = states[attnIndex] or {}
      local feats = states[featsIndex] or {}

      -- Ignore generated </s>.
      table.remove(tokens)
      if #attn > 0 then
        table.remove(attn)

        -- Remove unnecessary values from the attention vectors.
        if batch.size > 1 then
          local size = batch.sourceSize[b]
          local length = batch.sourceLength
          for j = 1, #attn do
            if length == attn[j]:size(1) then
              attn[j] = attn[j]:narrow(1, length - size + 1, size)
            end
          end
        end
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
    table.insert(allHistories, histories[b])
  end

  return allHyp, allFeats, allScores, allAttn, goldScore, allHistories
end

--[[ Save the aggreagated beam search histories in a JSON file.

See also https://github.com/OpenNMT/VisTools/blob/master/generate_beam_viz.py.

Parameters:

  * file - the file to save to.

]]
function Translator:saveBeamHistories(file)
  if not self.beamHistories or #self.beamHistories == 0 then
    return
  end

  local data = {}
  data.predicted_ids = {}
  data.beam_parent_ids = {}
  data.scores = {}

  for b = 1, #self.beamHistories do
    local history = self.beamHistories[b]
    local beamIds = {}
    local beamParents = {}
    local beamScores = {}

    for i = 1, #history.predictedIds do
      table.insert(beamIds,
                   self.dicts.tgt.words:convertToLabels(torch.totable(history.predictedIds[i])))
      table.insert(beamScores, torch.totable(history.scores[i]))
      table.insert(beamParents, torch.totable(history.parentBeams[i] - 1))
    end

    table.insert(data.predicted_ids, beamIds)
    table.insert(data.beam_parent_ids, beamParents)
    table.insert(data.scores, beamScores)
  end

  local output = assert(io.open(file, 'w'))
  output:write(require('dkjson').encode(data))
  self.beamHistories = {}
end

--[[ Translate a batch of source sequences.

Parameters:

  * `src` - a batch of tables containing:
    - `words`: the table of source words
    - `features`: the table of feaures sequences (`src.features[i][j]` is the value of the ith feature of the jth token)
  * `gold` - gold data to compute confidence score (same format as `src`)

Returns:

  * `results` - a batch of tables containing:
    - `goldScore`: if `gold` was given, this is the confidence score
    - `preds`: an array of `args.n_best` tables containing:
      - `words`: the table of target words
      - `features`: the table of target features sequences
      - `attention`: the attention vectors of each target word over the source words
      - `score`: the confidence score of the prediction
]]
function Translator:translate(src, gold)
  local data, ignored, indexMap = self:buildData(src, gold)

  local results = {}

  if data:batchCount() > 0 then
    local batch = onmt.utils.Cuda.convert(data:getBatch())

    local encStates = {}
    local pred = {}
    local predFeats = {}
    local predScore = {}
    local attn = {}
    local goldScore = {}
    local beamHistories = {}
    if self.args.dump_input_encoding then
      encStates = self:translateBatch(batch)
    else
      pred, predFeats, predScore, attn, goldScore, beamHistories = self:translateBatch(batch)
    end

    if self.beamHistories then
      onmt.utils.Table.append(self.beamHistories, beamHistories)
    end

    for b = 1, batch.size do
      if self.args.dump_input_encoding then
        results[b] = encStates[b]
      else
        results[b] = {}

        results[b].preds = {}
        for n = 1, self.args.n_best do
          results[b].preds[n] = {}
          results[b].preds[n].words = self:buildTargetWords(pred[b][n], src[indexMap[b]].words, attn[b][n])
          results[b].preds[n].features = self:buildTargetFeatures(predFeats[b][n])
          results[b].preds[n].attention = attn[b][n]
          results[b].preds[n].score = predScore[b][n]
        end
      end

      if goldScore and next(goldScore) ~= nil then
        results[b].goldScore = goldScore[b]
      end
    end
  end

  for i = 1, #ignored do
    table.insert(results, ignored[i], {})
  end

  return results
end

return Translator
