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

local bitextOptions = {
  {
    '-train_src', '',
    [[Path to the training source data.]],
    {
      valid = onmt.utils.ExtendedCmdLine.fileExists
    }
  },
  {
    '-train_tgt', '',
    [[Path to the training target data.]],
    {
      valid = onmt.utils.ExtendedCmdLine.fileExists
    }
  },
  {
    '-valid_src', '',
    [[Path to the validation source data.]],
    {
      valid = onmt.utils.ExtendedCmdLine.fileExists
    }
  },
  {
    '-valid_tgt', '',
    [[Path to the validation target data.]],
    {
      valid = onmt.utils.ExtendedCmdLine.fileExists
    }
  },
  {
    '-src_vocab', '',
    [[Path to an existing source vocabulary.]],
    {
      valid = onmt.utils.ExtendedCmdLine.fileNullOrExists
    }
  },
  {
    '-tgt_vocab', '',
    [[Path to an existing target vocabulary.]],
    {
      valid = onmt.utils.ExtendedCmdLine.fileNullOrExists
    }
  },
  {
    '-src_vocab_size', { 50000 },
    [[List of source vocabularies size: `word[ feat1[ feat2[ ...] ] ]`.
      If = 0, vocabularies are not pruned.]]
  },
  {
    '-tgt_vocab_size', { 50000 },
    [[List of target vocabularies size: `word[ feat1[ feat2[ ...] ] ]`.
      If = 0, vocabularies are not pruned.]]
  },
  {
    '-src_words_min_frequency', { 0 },
    [[List of source words min frequency: `word[ feat1[ feat2[ ...] ] ]`.
      If = 0, vocabularies are pruned by size.]]
  },
  {
    '-tgt_words_min_frequency', { 0 },
    [[List of target words min frequency: `word[ feat1[ feat2[ ...] ] ]`.
      If = 0, vocabularies are pruned by size.]]
  },
  {
    '-src_seq_length', 50,
    [[Maximum source sequence length.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt
    }
  },
  {
    '-tgt_seq_length', 50,
    [[Maximum target sequence length.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt
    }
  }
}

local monotextOptions = {
  {
    '-train', '',
    [[Path to the training source data.]],
    {
      valid = onmt.utils.ExtendedCmdLine.fileExists
    }
  },
  {
    '-valid', '',
    [[Path to the validation source data.]],
    {
      valid = onmt.utils.ExtendedCmdLine.fileExists
    }
  },
  {
    '-vocab', '',
    [[Path to an existing source vocabulary.]],
    {
      valid = onmt.utils.ExtendedCmdLine.fileNullOrExists
    }
  },
  {
    '-vocab_size', { 50000 },
    [[List of source vocabularies size: `word[ feat1[ feat2[ ...] ] ]`.
      If = 0, vocabularies are not pruned.]]
  },
  {
    '-words_min_frequency', { 0 },
    [[List of source words min frequency: `word[ feat1[ feat2[ ...] ] ]`.
      If = 0, vocabularies are pruned by size.]]
  },
  {
    '-seq_length', 50,
    [[Maximum source sequence length.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt()
    }
  }
}

local feattextOptions = {
  {
    '-train_src', '',
    [[Path to the training source data.]],
    {
      valid=onmt.utils.ExtendedCmdLine.fileExists
    }
  },
  {
    '-train_tgt', '',
    [[Path to the training target data.]],
    {
      valid=onmt.utils.ExtendedCmdLine.fileExists
    }
  },
  {
    '-valid_src', '',
    [[Path to the validation source data.]],
    {
      valid=onmt.utils.ExtendedCmdLine.fileExists
    }
  },
  {
    '-valid_tgt', '',
    [[Path to the validation target data.]],
    {
      valid=onmt.utils.ExtendedCmdLine.fileExists
    }
  },
  {
    '-tgt_vocab', '',
    [[Path to an existing target vocabulary.]],
    {
      valid=onmt.utils.ExtendedCmdLine.fileNullOrExists
    }
  },
  {
    '-tgt_vocab_size', { 50000 },
    [[List of target vocabularies size: word[ feat1[ feat2[ ...] ] ].
      If = 0, vocabularies are not pruned.]]
  },
  {
    '-tgt_words_min_frequency', { 0 },
    [[List of target words min frequency: word[ feat1[ feat2[ ...] ] ].
      If = 0, vocabularies are pruned by size.]]
  },
  {
    '-src_seq_length', 50,
    [[Maximum source sequence length.]],
    {
      valid=onmt.utils.ExtendedCmdLine.isUInt
    }
  },
  {
    '-tgt_seq_length', 50,
    [[Maximum target sequence length.]],
    {
      valid=onmt.utils.ExtendedCmdLine.isUInt
    }
  }
}

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
    '-sort', true,
    [[If = 1, sort the sentences by size to build batches without source padding.]]
  },
  {
    '-shuffle', true,
    [[If = 1, shuffle data (prior sorting).]]
  },
  {
    '-idx_files', false,
    [[If set, source and target files are 'key value' with key match between source and target.]]
  },
  {
    '-report_every', 100000,
    [[Report status every this many sentences.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt()
    }
  }
}

function Preprocessor.declareOpts(cmd, mode)
  mode = mode or 'bitext'
  local options
  if mode == 'bitext' then
    options = bitextOptions
  elseif mode == 'monotext' then
    options = monotextOptions
  else
    options = feattextOptions
  end
  for _, v in ipairs(commonOptions) do
    table.insert(options, v)
  end
  cmd:setCmdLineOptions(options, 'Data')
end

function Preprocessor:__init(args, mode)
  tds = require('tds')

  mode = mode or 'bitext'
  self.args = onmt.utils.ExtendedCmdLine.getModuleOpts(args, commonOptions)
  local options
  if mode == 'bitext' then
    options = bitextOptions
  elseif mode == 'monotext' then
    options = monotextOptions
  else
    options = feattextOptions
  end
  for _, v in ipairs(commonOptions) do
    table.insert(options, v)
  end
  self.args = onmt.utils.ExtendedCmdLine.getModuleOpts(args, options)
end

function Preprocessor:makeBilingualData(srcFile, tgtFile, srcDicts, tgtDicts, isValid)
    -- sentence length distribution
  local srcSentenceDist = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
  local tgtSentenceDist = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

  local src = tds.Vec()
  local srcFeatures = tds.Vec()

  local tgt = tds.Vec()
  local tgtFeatures = tds.Vec()

  local sizes = tds.Vec()

  local count = 0
  local ignored = 0
  local avgSrcLength = 0
  local avgTgtLength = 0
  local prunedRatioSrc = 0
  local prunedRatioTgt = 0


  local function processBilingualSentence(srcTokens, tgtTokens)
    local idxRange = math.floor(#srcTokens/10)+1
    if idxRange > #srcSentenceDist then
      idxRange = #srcSentenceDist
    end
    srcSentenceDist[idxRange] = srcSentenceDist[idxRange]+1
    idxRange = math.floor(#tgtTokens/10)+1
    if idxRange > #tgtSentenceDist then
      idxRange = #tgtSentenceDist
    end
    tgtSentenceDist[idxRange] = tgtSentenceDist[idxRange]+1

    if isValid(srcTokens, self.args.src_seq_length) and isValid(tgtTokens, self.args.tgt_seq_length) then
      avgSrcLength = avgSrcLength * (#src / (#src + 1)) + #srcTokens / (#src + 1)
      avgTgtLength = avgTgtLength * (#tgt / (#tgt + 1)) + #tgtTokens / (#tgt + 1)

      local srcWords, srcFeats = onmt.utils.Features.extract(srcTokens)
      local tgtWords, tgtFeats = onmt.utils.Features.extract(tgtTokens)

      local srcVec = srcDicts.words:convertToIdx(srcWords, onmt.Constants.UNK_WORD)
      local tgtVec = tgtDicts.words:convertToIdx(tgtWords,
                                                 onmt.Constants.UNK_WORD,
                                                 onmt.Constants.BOS_WORD,
                                                 onmt.Constants.EOS_WORD)

      local srcPruned = srcVec:eq(onmt.Constants.UNK):sum() / srcVec:size(1)
      local tgtPruned = tgtVec:eq(onmt.Constants.UNK):sum() / tgtVec:size(1)

      prunedRatioSrc = prunedRatioSrc * (#src / (#src + 1)) + srcPruned / (#src + 1)
      prunedRatioTgt = prunedRatioTgt * (#tgt / (#tgt + 1)) + tgtPruned / (#tgt + 1)

      src:insert(srcVec)
      tgt:insert(tgtVec)

      if #srcDicts.features > 0 then
        srcFeatures:insert(onmt.utils.Features.generateSource(srcDicts.features, srcFeats, true))
      end
      if #tgtDicts.features > 0 then
        tgtFeatures:insert(onmt.utils.Features.generateTarget(tgtDicts.features, tgtFeats, true, self.args.time_shift_feature))
      end

      sizes:insert(#srcWords)
    else
      ignored = ignored + 1
    end

    count = count + 1

    if count % self.args.report_every == 0 then
      _G.logger:info('... ' .. count .. ' sentences prepared')
    end
  end

  local srcReader = onmt.utils.FileReader.new(srcFile, self.args.idx_files)
  local tgtReader = onmt.utils.FileReader.new(tgtFile, self.args.idx_files)

  if self.args.idx_files then
    local srcDict = {}
    local srcCount = 0
    local tgtDict = {}
    local tgtCount = 0
    while true do
      local srcTokens, srcIdx = srcReader:next()
      if not srcTokens then
        break
      end
      if srcDict[srcIdx] then
        _G.logger:error('duplicate idx in src file: '..srcIdx)
        os.exit(1)
      end
      srcDict[srcIdx] = srcTokens
      srcCount = srcCount + 1
    end
    while true do
      local tgtTokens, tgtIdx = tgtReader:next()
      if not tgtTokens then
        break
      end
      if tgtDict[tgtIdx] then
        _G.logger:error('duplicate idx in src file: '..tgtIdx)
        os.exit(1)
      end
      if not srcDict[tgtIdx] then
        _G.logger:error('tgt Idx not defined in source: '..tgtIdx)
        os.exit(1)
      end
      tgtDict[tgtIdx] = tgtTokens
      tgtCount = tgtCount + 1
    end
    if srcCount ~= tgtCount then
      _G.logger:error('source Idx and target Idx not aligned')
      os.exit(1)
    end
    for k,v in pairs(srcDict) do
      processBilingualSentence(v, tgtDict[k])
    end
  else
    while true do
      local srcTokens = srcReader:next()
      local tgtTokens = tgtReader:next()

      if srcTokens == nil or tgtTokens == nil then
        if srcTokens == nil and tgtTokens ~= nil or srcTokens ~= nil and tgtTokens == nil then
          _G.logger:error('source and target do not have the same number of sentences')
          os.exit(1)
        end
        break
      end
      processBilingualSentence(srcTokens, tgtTokens)
    end
  end

  for i=1, #srcSentenceDist do
    srcSentenceDist[i] = srcSentenceDist[i]/count
    tgtSentenceDist[i] = tgtSentenceDist[i]/count
  end

  srcReader:close()
  tgtReader:close()

  local function reorderData(perm)
    src = onmt.utils.Table.reorder(src, perm, true)
    tgt = onmt.utils.Table.reorder(tgt, perm, true)

    if #srcDicts.features > 0 then
      srcFeatures = onmt.utils.Table.reorder(srcFeatures, perm, true)
    end
    if #tgtDicts.features > 0 then
      tgtFeatures = onmt.utils.Table.reorder(tgtFeatures, perm, true)
    end
  end

  if self.args.shuffle then
    _G.logger:info('... shuffling sentences')
    local perm = torch.randperm(#src)
    sizes = onmt.utils.Table.reorder(sizes, perm, true)
    reorderData(perm)
  end

  if self.args.sort then
    _G.logger:info('... sorting sentences by size')
    local _, perm = torch.sort(vecToTensor(sizes))
    reorderData(perm)
  end

  _G.logger:info('Prepared %d sentences:', #src)
  _G.logger:info(' * %d sequences ignored due to source length > %d or target length > %d',
                 ignored,
                 self.args.src_seq_length,
                 self.args.tgt_seq_length)
  _G.logger:info(' * average sequence length: source = %.1f, target = %.1f',
                 avgSrcLength,
                 avgTgtLength)
  _G.logger:info(' * %% of unkown words: source = %.1f%%, target = %.1f%%',
                 prunedRatioSrc * 100,
                 prunedRatioTgt * 100)

  local dist='[ '
  for i=1,#srcSentenceDist do
    if i>1 then
      dist = dist..' ; '
    end
    dist = dist..math.floor(srcSentenceDist[i]*100)..'%'
  end
  dist = dist..' ]'
  _G.logger:info(' * Source Sentence Length (range of 10): '..dist)
  dist='[ '
  for i=1,#tgtSentenceDist do
    if i>1 then
      dist = dist..' ; '
    end
    dist = dist..math.floor(tgtSentenceDist[i]*100)..'%'
  end
  dist = dist..' ]'
  _G.logger:info(' * Target Sentence Length (range of 10): '..dist)

  local srcData = {
    words = src,
    features = srcFeatures
  }

  local tgtData = {
    words = tgt,
    features = tgtFeatures
  }

  return srcData, tgtData
end

function Preprocessor:makeFeatTextData(srcFile, tgtFile, tgtDicts, isValid)
  local src = tds.Vec()
  local srcFeatures = tds.Vec()

  local tgt = tds.Vec()
  local tgtFeatures = tds.Vec()

  local sizes = tds.Vec()

  local count = 0
  local ignored = 0

  local function processFeatTextSentence(srcFeats, tgtTokens)
    if isValid(srcFeats, self.args.src_seq_length) and isValid(tgtTokens, self.args.tgt_seq_length) then
      local tgtWords, tgtFeats = onmt.utils.Features.extract(tgtTokens)

      src:insert(srcFeats)
      tgt:insert(tgtDicts.words:convertToIdx(tgtWords,
                                             onmt.Constants.UNK_WORD,
                                             onmt.Constants.BOS_WORD,
                                             onmt.Constants.EOS_WORD))

      if #tgtDicts.features > 0 then
        tgtFeatures:insert(onmt.utils.Features.generateTarget(tgtDicts.features, tgtFeats, true))
      end
      sizes:insert(srcFeats:size(1))
    else
      ignored = ignored + 1
    end

    count = count + 1

    if count % self.args.report_every == 0 then
      _G.logger:info('... ' .. count .. ' sequences prepared')
    end
  end

  local srcReader = onmt.utils.FileReader.new(srcFile, true, true)
  local tgtReader = onmt.utils.FileReader.new(tgtFile, true)

  if self.args.idx_files then
    local srcDict = {}
    local srcCount = 0
    local tgtDict = {}
    local tgtCount = 0
    while true do
      local srcFeats, srcIdx = srcReader:next()
      if not srcFeats then
        break
      end
      if srcDict[srcIdx] then
        _G.logger:error('duplicate idx in src file: '..srcIdx)
        os.exit(1)
      end
      srcDict[srcIdx] = torch.Tensor(srcFeats)
      srcCount = srcCount + 1
    end
    while true do
      local tgtTokens, tgtIdx = tgtReader:next()
      if not tgtTokens then
        break
      end
      if tgtDict[tgtIdx] then
        _G.logger:error('duplicate idx in src file: '..tgtIdx)
        os.exit(1)
      end
      if not srcDict[tgtIdx] then
        _G.logger:error('tgt Idx not defined in source: '..tgtIdx)
        os.exit(1)
      end
      tgtDict[tgtIdx] = tgtTokens
      tgtCount = tgtCount + 1
    end
    if srcCount ~= tgtCount then
      _G.logger:warning('missing '..(srcCount-tgtCount)..' sentences in target')
    end
    for k,v in pairs(srcDict) do
      if tgtDict[k] then
        processFeatTextSentence(v, tgtDict[k])
      end
    end
  else
    while true do
      local srcFeats = srcReader:next()
      local tgtTokens = tgtReader:next()

      if srcFeats == nil or tgtTokens == nil then
        if srcFeats == nil and tgtTokens ~= nil or srcFeats ~= nil and tgtTokens == nil then
          _G.logger:error('source and target do not have the same number of sentences')
          os.exit(1)
        end
        break
      end
      processFeatTextSentence(srcFeats, tgtTokens)
    end
  end

  srcReader:close()
  tgtReader:close()

  local function reorderData(perm)
    src = onmt.utils.Table.reorder(src, perm, true)
    tgt = onmt.utils.Table.reorder(tgt, perm, true)

    if #tgtDicts.features > 0 then
      tgtFeatures = onmt.utils.Table.reorder(tgtFeatures, perm, true)
    end
  end

  if self.args.shuffle then
    _G.logger:info('... shuffling sentences')
    local perm = torch.randperm(#src)
    sizes = onmt.utils.Table.reorder(sizes, perm, true)
    reorderData(perm)
  end

  if self.args.sort then
    _G.logger:info('... sorting sentences by size')
    local _, perm = torch.sort(vecToTensor(sizes))
    reorderData(perm)
  end

  _G.logger:info('Prepared ' .. #src .. ' sequences (' .. ignored
                   .. ' ignored due to source length > ' .. self.args.src_seq_length
                   .. ' or target length > ' .. self.args.tgt_seq_length .. ')')

  local srcData = {
    vectors = src,
    features = srcFeatures
  }

  local tgtData = {
    words = tgt,
    features = tgtFeatures
  }

  return srcData, tgtData
end

function Preprocessor:makeMonolingualData(file, dicts, isValid)
  local dataset = tds.Vec()
  local features = tds.Vec()

  local sizes = tds.Vec()

  local count = 0
  local ignored = 0
  local avgLength = 0
  local prunedRatio = 0

  local reader = onmt.utils.FileReader.new(file)

  while true do
    local tokens = reader:next()

    if tokens == nil then
      break
    end

    if isValid(tokens, self.args.seq_length) then
      avgLength = avgLength * (#dataset / (#dataset + 1)) + #tokens / (#dataset + 1)

      local words, feats = onmt.utils.Features.extract(tokens)
      local vec = dicts.words:convertToIdx(words, onmt.Constants.UNK_WORD)
      local pruned = vec:eq(onmt.Constants.UNK):sum() / vec:size(1)

      prunedRatio = prunedRatio * (#dataset / (#dataset + 1)) + pruned / (#dataset + 1)

      dataset:insert(vec)

      if #dicts.features > 0 then
        features:insert(onmt.utils.Features.generateSource(dicts.features, feats, true))
      end

      sizes:insert(#words)
    else
      ignored = ignored + 1
    end

    count = count + 1

  end

  reader:close()

  local function reorderData(perm)
    dataset = onmt.utils.Table.reorder(dataset, perm, true)

    if #dicts.features > 0 then
      features = onmt.utils.Table.reorder(features, perm, true)
    end
  end

  if self.args.shuffle then
    _G.logger:info('... shuffling sentences')
    local perm = torch.randperm(#dataset)
    sizes = onmt.utils.Table.reorder(sizes, perm, true)
    reorderData(perm)
  end

  if self.args.sort then
    _G.logger:info('... sorting sentences by size')
    local _, perm = torch.sort(vecToTensor(sizes))
    reorderData(perm)
  end

  _G.logger:info('Prepared %d sentences:', #dataset)
  _G.logger:info(' * %d sequences ignored due to length > %d', ignored, self.args.seq_length)
  _G.logger:info(' * average sequence length = %.1f', avgLength)
  _G.logger:info(' * %% of unkown words = %.1f%%', prunedRatio * 100)

  local data = {
    words = dataset,
    features = features
  }

  return data
end

return Preprocessor
