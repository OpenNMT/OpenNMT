--[[ Data Preparation functions. ]]

local function vecToTensor(vec)
  local t = torch.Tensor(#vec)
  for i, v in pairs(vec) do
    t[i] = v
  end
  return t
end

local Preprocessor = torch.class('Preprocessor')
local tds

local bitextOptions = {
  {'-train_src',               '',     [[Path to the training source data.]],
                                       {valid=onmt.utils.ExtendedCmdLine.fileExists}},
  {'-train_tgt',               '',     [[Path to the training target data.]],
                                       {valid=onmt.utils.ExtendedCmdLine.fileExists}},
  {'-valid_src',               '',     [[Path to the validation source data.]],
                                       {valid=onmt.utils.ExtendedCmdLine.fileExists}},
  {'-valid_tgt',               '',     [[Path to the validation target data.]],
                                       {valid=onmt.utils.ExtendedCmdLine.fileExists}},
  {'-src_vocab',               '',     [[Path to an existing source vocabulary.]],
                                       {valid=onmt.utils.ExtendedCmdLine.fileNullOrExists}},
  {'-tgt_vocab',               '',     [[Path to an existing target vocabulary.]],
                                       {valid=onmt.utils.ExtendedCmdLine.fileNullOrExists}},
  {'-src_vocab_size',          '50000',[[Comma-separated list of source vocabularies size: word[,feat1,feat2,...]. If = 0, vocabularies are not pruned.]],
                                       {valid=onmt.utils.ExtendedCmdLine.listUInt}},
  {'-tgt_vocab_size',          '50000',[[Comma-separated list of target vocabularies size: word[,feat1,feat2,...]. If = 0, vocabularies are not pruned.]],
                                       {valid=onmt.utils.ExtendedCmdLine.listUInt}},
  {'-src_words_min_frequency', '0',    [[Comma-separated list of source words min frequency: word[,feat1,feat2,...]. If = 0, vocabularies are pruned by size.]],
                                       {valid=onmt.utils.ExtendedCmdLine.listUInt}},
  {'-tgt_words_min_frequency', '0',    [[Comma-separated list of target words min frequency: word[,feat1,feat2,...]. If = 0, vocabularies are pruned by size.]],
                                       {valid=onmt.utils.ExtendedCmdLine.listUInt}},
  {'-src_seq_length',          50,     [[Maximum source sequence length.]],
                                       {valid=onmt.utils.ExtendedCmdLine.isUInt}},
  {'-tgt_seq_length',          50,     [[Maximum target sequence length.]],
                                       {valid=onmt.utils.ExtendedCmdLine.isUInt}}
}

local monotextOptions = {
  {'-train',                   '',     [[Path to the training source data.]],
                                       {valid=onmt.utils.ExtendedCmdLine.fileExists}},
  {'-valid',                   '',     [[Path to the validation source data.]],
                                       {valid=onmt.utils.ExtendedCmdLine.fileExists}},
  {'-vocab',                   '',     [[Path to an existing source vocabulary.]],
                                       {valid=onmt.utils.ExtendedCmdLine.fileNullOrExists}},
  {'-vocab_size',             '50000', [[Comma-separated list of source vocabularies size: word[,feat1,feat2,...]. If = 0, vocabularies are not pruned.]],
                                       {valid=onmt.utils.ExtendedCmdLine.listUInt}},
  {'-words_min_frequency',    '0',    [[Comma-separated list of source words min frequency: word[,feat1,feat2,...]. If = 0, vocabularies are pruned by size.]],
                                       {valid=onmt.utils.ExtendedCmdLine.listUInt}},
  {'-seq_length',              50,     [[Maximum source sequence length.]],
                                       {valid=onmt.utils.ExtendedCmdLine.isUInt()}}
}

local commonOptions = {
  {'-features_vocabs_prefix', '',      [[Path prefix to existing features vocabularies.]]},
  {'-shuffle',                1,       [[If 1, shuffle data.]],
                                       { valid=onmt.utils.ExtendedCmdLine.isInt(0,1)} }
}

function Preprocessor.declareOpts(cmd, mode)
  mode = mode or 'bitext'
  local options
  if mode == 'bitext' then
    options = bitextOptions
  else
    options = monotextOptions
  end
  for _, v in ipairs(commonOptions) do
    table.insert(options, v)
  end
  cmd:setCmdLineOptions(options, 'Preprocess')
end

function Preprocessor:__init(args, mode)
  tds = require('tds')

  mode = mode or 'bitext'
  local options
  if mode == 'bitext' then
    options = bitextOptions
  else
    options = monotextOptions
  end
  for _, v in ipairs(commonOptions) do
    table.insert(options, v)
  end
  self.args = onmt.utils.ExtendedCmdLine.getModuleOpts(args, options)
  self.args.report_every = args.report_every
end

function Preprocessor:makeBilingualData(srcFile, tgtFile, srcDicts, tgtDicts, isValid)
  local src = tds.Vec()
  local srcFeatures = tds.Vec()

  local tgt = tds.Vec()
  local tgtFeatures = tds.Vec()

  local sizes = tds.Vec()

  local count = 0
  local ignored = 0

  local srcReader = onmt.utils.FileReader.new(srcFile)
  local tgtReader = onmt.utils.FileReader.new(tgtFile)

  while true do
    local srcTokens = srcReader:next()
    local tgtTokens = tgtReader:next()

    if srcTokens == nil or tgtTokens == nil then
      if srcTokens == nil and tgtTokens ~= nil or srcTokens ~= nil and tgtTokens == nil then
        _G.logger:warning('source and target do not have the same number of sentences')
      end
      break
    end

    if isValid(srcTokens, self.args.src_seq_length) and isValid(tgtTokens, self.args.tgt_seq_length) then
      local srcWords, srcFeats = onmt.utils.Features.extract(srcTokens)
      local tgtWords, tgtFeats = onmt.utils.Features.extract(tgtTokens)

      src:insert(srcDicts.words:convertToIdx(srcWords, onmt.Constants.UNK_WORD))
      tgt:insert(tgtDicts.words:convertToIdx(tgtWords,
                                             onmt.Constants.UNK_WORD,
                                             onmt.Constants.BOS_WORD,
                                             onmt.Constants.EOS_WORD))

      if #srcDicts.features > 0 then
        srcFeatures:insert(onmt.utils.Features.generateSource(srcDicts.features, srcFeats, true))
      end
      if #tgtDicts.features > 0 then
        tgtFeatures:insert(onmt.utils.Features.generateTarget(tgtDicts.features, tgtFeats, true))
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

  if self.args.shuffle == 1 then
    _G.logger:info('... shuffling sentences')
    local perm = torch.randperm(#src)
    sizes = onmt.utils.Table.reorder(sizes, perm, true)
    reorderData(perm)
  end

  _G.logger:info('... sorting sentences by size')
  local _, perm = torch.sort(vecToTensor(sizes))
  reorderData(perm)

  _G.logger:info('Prepared ' .. #src .. ' sentences (' .. ignored
                   .. ' ignored due to source length > ' .. self.args.src_seq_length
                   .. ' or target length > ' .. self.args.tgt_seq_length .. ')')

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

function Preprocessor:makeMonolingualData(file, dicts, isValid)
  local dataset = tds.Vec()
  local features = tds.Vec()

  local sizes = tds.Vec()

  local count = 0
  local ignored = 0

  local reader = onmt.utils.FileReader.new(file)

  while true do
    local tokens = reader:next()

    if tokens == nil then
      break
    end

    if isValid(tokens, self.args.seq_length) then
      local words, feats = onmt.utils.Features.extract(tokens)

      dataset:insert(dicts.words:convertToIdx(words, onmt.Constants.UNK_WORD))

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

  if self.args.shuffle == 1 then
    _G.logger:info('... shuffling sentences')
    local perm = torch.randperm(#dataset)
    sizes = onmt.utils.Table.reorder(sizes, perm, true)
    reorderData(perm)
  end

  _G.logger:info('... sorting sentences by size')
  local _, perm = torch.sort(vecToTensor(sizes))
  reorderData(perm)

  _G.logger:info('Prepared ' .. #dataset .. ' sentences (' .. ignored
                   .. ' ignored due to length > ' .. self.args.seq_length .. ')')

  local data = {
    words = dataset,
    features = features
  }

  return data
end

return Preprocessor
