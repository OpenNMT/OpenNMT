local Translator = torch.class('Translator')

local options = {
  {
    '-model', '',
    [[Path to the serialized model file.]],
    {
      valid = onmt.utils.ExtendedCmdLine.fileNullOrExists
    }
  },
  {
    '-lm_model', '',
    [[Path to serialized language model file.]],
    {
      valid = onmt.utils.ExtendedCmdLine.fileNullOrExists
    }
  },
  {
    '-lm_weight', 0.1,
    [[Relative weight of language model.]]
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
    '-replace_unk_tagged', false,
    [[The same as -replace_unk, but wrap the replaced token in ｟unk:xxxxx｠ if it is not found in the phrase table.]]},
  {
    '-lexical_constraints', false,
    [[Force the beam search to apply the translations from the phrase table.]]
  },
  {
    '-limit_lexical_constraints', false,
    [[Prevents producing each lexical constraint more than required.]]
  },
  {
    '-placeholder_constraints', false,
    [[Force the beam search to reproduce placeholders in the translation.]]
  },
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

function Translator.expandOpts(cmd, dataType)
  local torenameOpts = {};
  local current_block;
  local pref = "{src,tgt}_"
  if dataType == "monotext" then pref = "" end
  if dataType == "feattext" then pref = "tgt_" end
  for i, v in ipairs(cmd.helplines) do
    if type(v) == "string" then
      local p = v:find(" options")
      if p then
        current_block = v:sub(1,p-1);
        if current_block == "MPreprocessing" or current_block == "Tokenizer" then
          cmd.helplines[i] = cmd.helplines[i]
        end
      end
    else
      if current_block == "MPreprocessing" or current_block == "Tokenizer" then
        torenameOpts[v.key] = current_block:sub(1,3):lower()
        v.key="-"..current_block:sub(1,3):lower().."_"..pref..v.key:sub(2)
      end
    end
  end

  cmd.options['-mode'] =
    {
      type= "string",
      key= cmd.options['-mode'].key,
      default= 'space',
      help= [[Define how aggressive should the tokenization be. `space` is space-tokenization.]],
      meta= {
          enum = {'conservative', 'aggressive', 'space'}
      }
    }

  local newOpts = {}
  for k, v in pairs(cmd.options) do
    if torenameOpts[k] then
      cmd.options[k] = nil
      if dataType == 'monotext' then
        local ksrc = '-'..torenameOpts[k]..'_'..k:sub(2)
        newOpts[ksrc] = onmt.utils.Table.deepCopy(v)
      elseif dataType == 'bitext' then
        local ksrc = '-'..torenameOpts[k]..'_src_'..k:sub(2)
        newOpts[ksrc] = onmt.utils.Table.deepCopy(v)
      end
      if dataType ~= 'monotext' then
        local ktgt = '-'..torenameOpts[k]..'_tgt_'..k:sub(2)
        newOpts[ktgt] = onmt.utils.Table.deepCopy(v)
      end
    end
  end
  for k, v in pairs(newOpts) do
    cmd.options[k] = v
  end
end

function Translator:__init(args, model, dicts)
  self.args = args

  if model then
    self.model = model
    self.dicts = dicts
  else
    _G.logger:info('Loading \'' .. self.args.model .. '\'...')
    local checkpoint = torch.load(self.args.model)

    self.dataType = checkpoint.options.data_type or 'bitext'
    self.modelType = checkpoint.options.model_type or 'seq2seq'
    _G.logger:info('Model %s trained on %s', self.modelType, self.dataType)

    onmt.utils.Error.assert(self.modelType == 'seq2seq', "Translator can only manage seq2seq models")

    self.model = onmt.Seq2Seq.load(args, checkpoint.models, checkpoint.dicts)
    self.dicts = checkpoint.dicts
  end

  self.model:evaluate()
  onmt.utils.Cuda.convert(self.model.models)

  -- TODO : extend phrase table to phrases with several words
  if self.args.phrase_table:len() > 0 then
    self.phraseTable = onmt.translate.PhraseTable.new(self.args.phrase_table)
  end

  if args.limit_lexical_constraints and args.placeholder_constraints then
    self.placeholderMask = self.dicts.tgt.words:getPlaceholderMask()
  end

  if args.lm_model ~= '' then
    local tmodel = args.model
    args.model = args.lm_model
    self.lm = onmt.lm.LM.new(args)

    -- check that lm has the same dictionary than translation model
    onmt.utils.Error.assert(self.dicts.tgt.words == self.lm.dicts.src.words, "Language model dictionary does not match seq2seq target dictionary")
    onmt.utils.Error.assert(#self.dicts.tgt.features == #self.lm.dicts.src.features, "Language model should have same number of features than translation model")
    for i = 1 , #self.dicts.tgt.features do
      onmt.utils.Error.assert(self.dicts.tgt.features[i] == self.lm.dicts.src.features[i], "LM feature ["..i.."] does not match translation tgt feature")
    end

    args.model = tmodel
  end

  if self.args.target_subdict:len() > 0 then
    self.dicts.subdict = onmt.utils.SubDict.new(self.dicts.tgt.words, self.args.target_subdict)
    self.model:setGeneratorVocab(self.dicts.subdict.targetVocTensor)
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
    local vocabs, placeholders = onmt.utils.Placeholders.norm(words)

    data.words = vocabs
    data.placeholders = placeholders

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
  srcData.constraints = {}

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
    if (src[b].words and #src[b].words == 0
        or src[b].vectors and src[b].vectors:dim() == 0) then
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

        if self.args.placeholder_constraints then
          local c = {}
          for ph,_ in pairs(src[b].placeholders) do
            if (self.dicts.tgt.words:lookup(ph)) then
              table.insert(c, ph)
            end
          end
          table.insert(srcData.constraints, self.dicts.tgt.words:convertToIdx(c, onmt.Constants.UNK_WORD))
        end

        if self.phraseTable and self.args.lexical_constraints then
          local c = {}
          for _,w in pairs(src[b].words) do
            if (self.phraseTable:contains(w)) then
              -- TODO : deal with phrases and source words
              local tgt = self.phraseTable:lookup(w)
              if (self.dicts.tgt.words:lookup(tgt)) then
                table.insert(c, tgt)
              end
            end
          end
          table.insert(srcData.constraints, self.dicts.tgt.words:convertToIdx(c, onmt.Constants.UNK_WORD))
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

function Translator:buildTargetWords(pred, src, attn, placeholders)
  local tokens = self.dicts.tgt.words:convertToLabels(pred, onmt.Constants.EOS)

  if self.args.replace_unk or self.args.replace_unk_tagged or self.args.placeholder_constraints then
    for i = 1, #tokens do
      if tokens[i] == onmt.Constants.UNK_WORD and (self.args.replace_unk or self.args.replace_unk_tagged) then
        local _, maxIndex = attn[i]:max(1)
        local source = src[maxIndex[1]]

        if self.phraseTable and self.phraseTable:contains(source) then
          tokens[i] = self.phraseTable:lookup(source)

        elseif self.args.replace_unk then
          tokens[i] = source

        elseif self.args.replace_unk_tagged then
          tokens[i] = '｟unk:' .. source .. '｠'
        end
      end
      if placeholders[tokens[i]] and self.args.placeholder_constraints then
        tokens[i] = placeholders[tokens[i]]
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
  local encStates, context = self.model.models.encoder:forward(batch)
  if self.args.dump_input_encoding then
    return encStates[#encStates]
  end

  -- if we have a language model - initialize lm state with BOS
  local lmStates, lmContext
  if self.lm then
    local bos_inputs = torch.IntTensor(batch.size):fill(onmt.Constants.BOS)
    if #self.lm.dicts.src.features > 0 then
      local inputs = { bos_inputs }
      if #self.lm.dicts.src.features > 1 then
        table.insert(inputs, {})
        for _ = 1, #self.lm.dicts.src.features do
          table.insert(inputs[2], bos_inputs)
        end
      else
        table.insert(inputs, bos_inputs)
      end
      bos_inputs = inputs
    end
    onmt.utils.Cuda.convert(bos_inputs)
    lmStates, lmContext = self.lm.model.models.encoder:forwardOne(bos_inputs, nil, true)
  end

  local decInitStates = self.model.models.bridge:forward(encStates)

  -- Compute gold score.
  local goldScore
  if batch.targetInput ~= nil then
    goldScore = self.model.models.decoder:computeScore(batch, decInitStates, context)
  end

  -- Specify how to go one step forward.
  local advancer = onmt.translate.DecoderAdvancer.new(self.model.models.decoder,
                                                      batch,
                                                      context,
                                                      self.args.max_sent_length,
                                                      self.args.max_num_unks,
                                                      decInitStates,
                                                      self.lm and self.lm.model.models,
                                                      lmStates, lmContext, self.args.lm_weight,
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
                                                 self.args.pre_filter_factor,
                                                 false,
                                                 self.placeholderMask)

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

  local output = onmt.utils.Error.assert(io.open(file, 'w'), "Cannot open file '"..file.."' for writing.")
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
          results[b].preds[n].words = self:buildTargetWords(pred[b][n], src[indexMap[b]].words, attn[b][n], src[indexMap[b]].placeholders)
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
