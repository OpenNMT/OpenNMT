--[[ Data Preparation functions. ]]

local function vecToTensor(vec)
  local t = torch.Tensor(#vec)
  for i = 1, #vec do
    t[i] = vec[i]
  end
  return t
end

local Preprocessor = torch.class('Preprocessor')
local tds

local commonOptions = {
  {
    '-features_vocabs_prefix', '',
    [[Path prefix to existing features vocabularies.]]
  },
  {
    '-time_shift_feature', true,
    [[Time shift features on the decoder side.]]
  },
  {
    '-keep_frequency', false,
    [[Keep frequency of words in dictionary.]]
  },
  {
    '-sample', 0,
    [[If not zero, extract a sample of the corpus. Values between 0 and 1 indicate ratio,
      values higher than 1 indicate data size]],
    {
      valid = onmt.utils.ExtendedCmdLine.isFloat(0)
    }
  },
  {
    '-sample_dist', '',
    [[Configuration file with data class distribution to use for sampling training corpus.]],
    {
      valid = onmt.utils.ExtendedCmdLine.fileNullOrExists,
      depends = function(opt) return opt.sample_dist == '' or opt.sample > 0 end
    }
  },
  {
    '-sort', true,
    [[If set, sort the sequences by size to build batches without source padding.]]
  },
  {
    '-shuffle', true,
    [[If set, shuffle the data (prior sorting).]]
  },
  {
    '-idx_files', false,
    [[If set, source and target files are 'key value' with key match between source and target.]]
  },
  {
    '-report_every', 100000,
    [[Report status every this many sentences.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isInt(1)
    }
  }
}

function Preprocessor.getDataList(dataType)
  local datalist
  if dataType == 'bitext' or dataType == 'seq2seq' then
    datalist = { {name="source",short="src",hasVocab=true,suffix=".src"} , {name="target",short="tgt",hasVocab=true,suffix=".tgt"} }
  elseif dataType == 'monotext' then
    datalist = { {hasVocab=true,suffix=".tok"} }
  else
    datalist = { {name="source",short="src",hasVocab=false,suffix=".src"} , {name="target",short="tgt",hasVocab=true,suffix=".tgt"} }
  end
  return datalist
end

--[[
  Generic function to generate options for the different dataTypes
]]
local function declareDataOptions(dataType)
  local function prefix(data)
    if data.short then return data.short.."_" end
    return ""
  end
  local function suffix(data)
    if data.short then return "_"..data.short end
    return ""
  end
  local function nameWithSpace(data)
    if data.name then return " "..data.name end
    return ""
  end
  local datalist = Preprocessor.getDataList(dataType)
  local options = {}

  table.insert(options,
    {
      '-train_dir', '',
      [[Path to training files directory.]],
      {
        valid = onmt.utils.ExtendedCmdLine.fileNullOrExists
      }
    })
  for i = 1, #datalist do
    table.insert(options,
      {
        '-train'..suffix(datalist[i]), '',
        "Path to the training"..nameWithSpace(datalist[i]).." data.",
        {
          valid=onmt.utils.ExtendedCmdLine.fileNullOrExists
        }
      })
  end

  for i = 1, #datalist do
    table.insert(options,
      {
        '-valid'..suffix(datalist[i]), '',
        "Path to the validation"..nameWithSpace(datalist[i]).." data.",
        {
          valid=onmt.utils.ExtendedCmdLine.fileExists
        }
      })
  end
  for i = 1, #datalist do
    if datalist[i].hasVocab then
      table.insert(options,
        {
          '-'..prefix(datalist[i])..'vocab', '',
          "Path to an existing"..nameWithSpace(datalist[i]).." vocabulary.",
          {
            valid=onmt.utils.ExtendedCmdLine.fileNullOrExists
          }
        })
      table.insert(options,
        {
          '-'..prefix(datalist[i])..'suffix', datalist[i].suffix,
          "Suffix for"..nameWithSpace(datalist[i]).." files in train/valid directories."
        })
      table.insert(options,
        {
          '-'..prefix(datalist[i])..'vocab_size', { 50000 },
          "List of"..nameWithSpace(datalist[i])..[[ vocabularies size: `word[ feat1[ feat2[ ...] ] ]`.
            If = 0, vocabularies are not pruned.]]
        })
      table.insert(options,
        {
          '-'..prefix(datalist[i])..'words_min_frequency', { 0 },
          "List of"..nameWithSpace(datalist[i])..[[ words min frequency: `word[ feat1[ feat2[ ...] ] ]`.
            If = 0, vocabularies are pruned by size.]]
        })
    end
  end
  for i = 1, #datalist do
    table.insert(options,
      {
        '-'..prefix(datalist[i])..'seq_length', 50,
        "Maximum"..nameWithSpace(datalist[i])..[[ sequence length.]],
        {
          valid = onmt.utils.ExtendedCmdLine.isInt(1)
        }
      })
  end

  if dataType ~= 'monotext' then
    table.insert(options,
      {
        '-check_plength', false,
        [[Check source and target have same length (for seq tagging).]]
      })
  end

  return options
end

function Preprocessor.declareOpts(cmd, dataType)
  dataType = dataType or 'bitext'
  local options = declareDataOptions(dataType)
  for _, v in ipairs(commonOptions) do
    table.insert(options, v)
  end
  cmd:setCmdLineOptions(options, 'Data')
end

function Preprocessor:__init(args, dataType)
  tds = require('tds')

  self.dataType = dataType or 'bitext'
  self.args = onmt.utils.ExtendedCmdLine.getModuleOpts(args, commonOptions)
  local options = declareDataOptions(self.dataType)
  for _, v in ipairs(commonOptions) do
    table.insert(options, v)
  end
  self.args = onmt.utils.ExtendedCmdLine.getModuleOpts(args, options)

  local function isempty(t)
    local count = 0
    for _, v in ipairs(t) do
      if self.args[v] == '' then
        count = count + 1
      end
    end
    return count
  end

  -- sanity check on options: train_dir is exclusive all direct file settings
  -- and for train_dir, we do need pre-build vocabulary
  if dataType == 'monotext' then
    self.trains = { 'train' }
    self.vocabs = { 'vocab' }
  else
    self.trains = { 'train_src', 'train_tgt' }
    self.vocabs = { 'src_vocab', 'tgt_vocab' }
  end

  if args.train_dir ~= '' then
    assert(isempty(self.trains) == #self.trains, 'For directory mode, file mode options (training) should not be set')
    assert(isempty(self.vocabs) == 0, 'For directory mode, vocabs should be predefined')
  else
    assert(isempty(self.trains) == 0)
  end
end

--[[
  Generic data preparation function on multiples source
  * `files`: table of data source name
  * `isInputVector`: table of boolean indicating if corresponding source is an vector
  * `dicts`: table of dictionary data corresponding to source
  * `nameSources`: table of name of each source - for logging purpose
  * `constants`: constant to add to the vocabulary for each source
  * `isValid`: validation function taking prepared table of tokens from each source
  * `generateFeatures`: table of feature extraction fucnction for each source
]]
function Preprocessor:makeGenericData(files, isInputVector, dicts, nameSources, constants, isValid, generateFeatures, parallelCheck)
  assert(#files==#dicts, "dict table should match files table")
  local n = #files
  local sentenceDists = {}
  local vectors = {}
  local features = {}
  local avgLength = {}
  local prunedRatio = {}
  local readers = {}
  local sizes = tds.Vec()

  for i = 1, n do
    table.insert(sentenceDists, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
    table.insert(vectors, tds.Vec())
    table.insert(features, tds.Vec())
    table.insert(avgLength, 0)
    table.insert(prunedRatio, 0)
    table.insert(readers, onmt.utils.FileReader.new(files[i], self.args.idx_files, isInputVector[i]))
  end

  local count = 0
  local ignored = 0
  local emptyCount = 0

  local function processSentence(idx, tokens)
    for i = 1, n do
      local length = (type(tokens[i])=='table' and #tokens[i]) or (tokens[i]:dim()==0 and 0) or tokens[i]:size(1)
      local idxRange = math.floor(length/10)+1
      if idxRange > #sentenceDists[i] then
        idxRange = #sentenceDists[i]
      end
      sentenceDists[i][idxRange] = sentenceDists[i][idxRange]+1
    end

    if parallelCheck then
      parallelCheck(idx, isInputVector, dicts, tokens)
    end

    if isValid(tokens) then
      for i = 1, n do
        local length = (type(tokens[i])=='table' and #tokens[i]) or (tokens[i]:dim()==0 and 0) or tokens[i]:size(1)
        avgLength[i] = avgLength[i] * (#vectors[i] / (#vectors[i] + 1)) + length / (#vectors[i] + 1)

        if isInputVector[i] then
          vectors[i]:insert(tokens[i])
        else
          local words, feats = onmt.utils.Features.extract(tokens[i])
          local vec = dicts[i].words:convertToIdx(words, table.unpack(constants[i]))
          local pruned = vec:eq(onmt.Constants.UNK):sum() / vec:size(1)

          prunedRatio[i] = prunedRatio[i] * (#vectors[i] / (#vectors[i] + 1)) + pruned / (#vectors[i] + 1)
          vectors[i]:insert(vec)

          if not(isInputVector[i]) and #dicts[i].features > 0 then
            features[i]:insert(generateFeatures[i](dicts[i].features, feats, true, self.args.time_shift_feature))
          end
        end

        if i == 1 then
          sizes:insert(length)
        end
      end
    else
      ignored = ignored + 1
    end

    count = count + 1

    if count % self.args.report_every == 0 then
      _G.logger:info('... ' .. count .. ' sentences prepared')
    end
  end

  if self.args.idx_files then
    local maps = {}
    for i = 1, n do
      table.insert(maps, {})
      while true do
        local tokens, idx = readers[i]:next()
        if not tokens then
          break
        end
        if maps[i][idx] then
          _G.logger:error('duplicate idx in %s file: '..idx, nameSources[i])
          os.exit(1)
        end
        if i > 1 and not maps[1][idx] then
          _G.logger:error('%s Idx not defined in %s: '..idx, nameSources[i], nameSources[1])
          os.exit(1)
        end
        if isInputVector[i] then
          maps[i][idx] = torch.Tensor(tokens)
        else
          maps[i][idx] = tokens
        end
      end
    end
    for k,_ in pairs(maps[1]) do
      local tokens = {}
      local hasNil = false
      for i = 1, n do
        hasNil = hasNil or maps[i][k] == nil
        table.insert(tokens, maps[i][k])
      end
      if not hasNil then
        processSentence(k, tokens)
      else
        emptyCount = emptyCount + 1
      end
    end
  else
    local idx = 1
    while true do
      local tokens = {}
      local hasNil = false
      local allNil = true
      for i = 1, n do
        tokens[i] = readers[i]:next()
        hasNil = hasNil or tokens[i] == nil
        allNil = allNil and tokens[i] == nil
      end

      if hasNil then
        if not allNil then
          _G.logger:error('all data sources do not have the same number of sentences')
          os.exit(1)
        end
        break
      end
      processSentence(idx, tokens)
      idx = idx + 1
    end
  end

  for i = 1, n do
    for j=1, #sentenceDists[i] do
      sentenceDists[i][j] = sentenceDists[i][j]/count
    end
    readers[i]:close()
  end

  local function reorderData(perm)
    for i = 1, n do
      vectors[i] = onmt.utils.Table.reorder(vectors[i], perm, true, isInputVector and isInputVector[i])
      if not(isInputVector[i]) and #dicts[i].features > 0 then
        features[i] = onmt.utils.Table.reorder(features[i], perm, true)
      end
    end
  end

  if self.args.shuffle then
    _G.logger:info('... shuffling sentences')
    local perm = torch.randperm(#vectors[1])
    sizes = onmt.utils.Table.reorder(sizes, perm, true)
    reorderData(perm)
  end

  if self.args.sort then
    _G.logger:info('... sorting sentences by size')
    local _, perm = torch.sort(vecToTensor(sizes))
    reorderData(perm)
  end

  _G.logger:info('Prepared %d sentences:', #vectors[1])
  _G.logger:info(' * %d sequences not validated (length, other)', ignored)
  local msgLength = ''
  local msgPrune = ''
  for i = 1, n do
    msgLength = msgLength .. (i==1 and '' or ', ')
    msgLength = msgLength .. nameSources[i] .. ' = '..string.format("%.1f", avgLength[i])
    msgPrune = msgPrune .. (i==1 and '' or ', ')
    msgPrune = msgPrune .. nameSources[i] .. ' = '..string.format("%.1f%%", prunedRatio[i] * 100)
  end

  _G.logger:info(' * average sequence length: '..msgLength)
  _G.logger:info(' * % of unknown words: '..msgPrune)

  local data = {}

  for i = 1, n do
    local dist='[ '
    for j = 1, #sentenceDists[1] do
      if j>1 then
        dist = dist..' ; '
      end
      dist = dist..math.floor(sentenceDists[i][j]*100)..'%%'
    end
    dist = dist..' ]'
    _G.logger:info(' * %s sentence length (range of 10): '..dist, nameSources[i])

    if isInputVector[i] then
      table.insert(data, { vectors = vectors[i], features = features[i] })
    else
      table.insert(data, { words = vectors[i], features = features[i] })
    end

  end

  return data
end

function Preprocessor:makeBilingualData(srcFile, tgtFile, srcDicts, tgtDicts, isValid, parallelCheck)
  local data = self:makeGenericData(
                              { srcFile, tgtFile },
                              { false, false },
                              { srcDicts, tgtDicts },
                              { 'source', 'target' },
                              {
                                {
                                  onmt.Constants.UNK_WORD
                                },
                                {
                                  onmt.Constants.UNK_WORD,
                                  onmt.Constants.BOS_WORD,
                                  onmt.Constants.EOS_WORD
                                }
                              },
                              function(tokens)
                                return #tokens[1] > 0 and
                                       isValid(tokens[1], self.args.src_seq_length) and
                                       #tokens[2] > 0 and
                                       isValid(tokens[2], self.args.tgt_seq_length)
                              end,
                              {
                                onmt.utils.Features.generateSource,
                                onmt.utils.Features.generateTarget
                              },
                              parallelCheck)
  return data[1], data[2]
end

function Preprocessor:makeFeatTextData(srcFile, tgtFile, tgtDicts, isValid, parallelCheck)
  local data = self:makeGenericData(
                              { srcFile, tgtFile },
                              { true, false },
                              { {}, tgtDicts },
                              { 'source', 'target' },
                              {
                                false,
                                {
                                  onmt.Constants.UNK_WORD,
                                  onmt.Constants.BOS_WORD,
                                  onmt.Constants.EOS_WORD
                                }
                              },
                              function(tokens)
                                return tokens[1]:dim() > 0 and
                                       isValid(tokens[1], self.args.src_seq_length) and
                                       #tokens[2] > 0 and
                                       isValid(tokens[2], self.args.tgt_seq_length)
                              end,
                              {
                                false,
                                onmt.utils.Features.generateTarget
                              },
                              parallelCheck)
  return data[1], data[2]
end

function Preprocessor:makeMonolingualData(file, dicts, isValid)
  local data = self:makeGenericData(
                              { file },
                              { false },
                              { dicts },
                              { 'source' },
                              {
                                {
                                  onmt.Constants.UNK_WORD,
                                  onmt.Constants.BOS_WORD,
                                  onmt.Constants.EOS_WORD
                                }
                              },
                              function(tokens)
                                return #tokens[1] > 0 and isValid(tokens[1], self.args.seq_length)
                              end,
                              {
                                onmt.utils.Features.generateTarget
                              })
  return data[1]
end

--[[ Check data validity ]]
function Preprocessor.isValid(seq, maxSeqLength)
  if torch.isTensor(seq) then
    return seq:size(1) > 0 and seq:size(1) <= maxSeqLength
  end
  return #seq > 0 and #seq <= maxSeqLength
end

local function parallelCheck(idx, _, _, tokens)
  local length1 = (type(tokens[1])=='table' and #tokens[1]) or (tokens[1]:dim()==0 and 0) or tokens[1]:size(1)
  local length2 = (type(tokens[2])=='table' and #tokens[2]) or (tokens[2]:dim()==0 and 0) or tokens[2]:size(1)
  if length1~=length2 then
    _G.logger:warning('SENT %s: source/target not aligned (%d/%d)', tostring(idx), length1, length2)
    return false
  end
  return true
end

function Preprocessor:getVocabulary()
  local dicts = {}
  if self.dataType ~= 'feattext' then
    local src_file = self.args.train_src
    if self.dataType == 'monotext' then
      src_file = self.args.train
    end
    dicts.src = onmt.data.Vocabulary.init('source',
                                     src_file,
                                     self.args.src_vocab or self.args.vocab,
                                     self.args.src_vocab_size or self.args.vocab_size,
                                     self.args.src_words_min_frequency or self.args.words_min_frequency,
                                     self.args.features_vocabs_prefix,
                                     function(s) return onmt.data.Preprocessor.isValid(s, self.args.src_seq_length or self.args.seq_length) end,
                                     self.args.keep_frequency,
                                     self.args.idx_files)
  end
  if self.dataType ~= 'monotext' then
    local tgt_file = self.args.train_tgt
    dicts.tgt = onmt.data.Vocabulary.init('target',
                                     tgt_file,
                                     self.args.tgt_vocab,
                                     self.args.tgt_vocab_size,
                                     self.args.tgt_words_min_frequency,
                                     self.args.features_vocabs_prefix,
                                     function(s) return onmt.data.Preprocessor.isValid(s, self.args.tgt_seq_length) end,
                                     self.args.keep_frequency,
                                     self.args.idx_files)
  end
  return dicts
end

function Preprocessor:makeData(dataset, dicts)
  local parallelValidFunc = nil
  if self.args.check_plength then
    parallelValidFunc = parallelCheck
  end

  local data = {}
  if self.dataType == 'monotext' then
    data.src = self:makeMonolingualData(self.args[dataset], dicts.src, self.isValid)
  elseif self.dataType == 'feattext' then
    data.src, data.tgt = self:makeFeatTextData(self.args[dataset.."_src"], self.args[dataset.."_tgt"],
                                               dicts.tgt,
                                               self.isValid, parallelValidFunc)
    if dataset == 'train' then
      -- record the size of the input layer
      dicts.srcInputSize = data.src.vectors[1]:size(2)
    end
  else
    data.src, data.tgt = self:makeBilingualData(self.args[dataset.."_src"], self.args[dataset.."_tgt"],
                                                dicts.src, dicts.tgt,
                                                self.isValid, parallelValidFunc)
  end

  return data
end

return Preprocessor
