require('onmt.init')

local tds = require('tds')
local path = require('pl.path')

local cmd = torch.CmdLine()

cmd:text("")
cmd:text("preprocess.lua")
cmd:text("")
cmd:text("**Preprocess Options**")
cmd:text("")
cmd:text("")
cmd:option('-config', '', [[Read options from this file]])

cmd:option('-train_src', '', [[Path to the training source data]])
cmd:option('-train_tgt', '', [[Path to the training target data]])
cmd:option('-valid_src', '', [[Path to the validation source data]])
cmd:option('-valid_tgt', '', [[Path to the validation target data]])

cmd:option('-save_data', '', [[Output file for the prepared data]])

cmd:option('-src_vocab_size', 50000, [[Size of the source vocabulary]])
cmd:option('-tgt_vocab_size', 50000, [[Size of the target vocabulary]])
cmd:option('-src_vocab', '', [[Path to an existing source vocabulary]])
cmd:option('-tgt_vocab', '', [[Path to an existing target vocabulary]])
cmd:option('-features_vocabs_prefix', '', [[Path prefix to existing features vocabularies]])

cmd:option('-src_seq_length', 50, [[Maximum source sequence length]])
cmd:option('-tgt_seq_length', 50, [[Maximum target sequence length]])
cmd:option('-shuffle', 1, [[Shuffle data]])
cmd:option('-seed', 3435, [[Random seed]])

cmd:option('-report_every', 100000, [[Report status every this many sentences]])
cmd:option('-log_file', '', [[Outputs logs to a file under this path instead of stdout]])
cmd:option('-disable_logs', false, [[Outputs nothing]])

local opt = cmd:parse(arg)

local function hasFeatures(filename)
  local reader = onmt.utils.FileReader.new(filename)
  local _, _, numFeatures = onmt.utils.Features.extract(reader:next())
  reader:close()
  return numFeatures > 0
end

local function isValid(sent, maxSeqLength)
  return #sent > 0 and #sent <= maxSeqLength
end

local function makeVocabulary(filename, size, maxSeqLength)
  local wordVocab = onmt.utils.Dict.new({onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                                         onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD})
  local featuresVocabs = {}

  local reader = onmt.utils.FileReader.new(filename)

  while true do
    local sent = reader:next()
    if sent == nil then
      break
    end

    if isValid(sent, maxSeqLength) then
      local words, features, numFeatures = onmt.utils.Features.extract(sent)

      if #featuresVocabs == 0 and numFeatures > 0 then
        for j = 1, numFeatures do
          featuresVocabs[j] = onmt.utils.Dict.new({onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                                                   onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD})
        end
      else
        assert(#featuresVocabs == numFeatures,
               'all sentences must have the same numbers of additional features')
      end

      for i = 1, #words do
        wordVocab:add(words[i])

        for j = 1, numFeatures do
          featuresVocabs[j]:add(features[j][i])
        end
      end
    end

  end

  reader:close()

  local originalSize = wordVocab:size()
  wordVocab = wordVocab:prune(size)
  _G.logger:info('Created dictionary of size ' .. wordVocab:size() .. ' (pruned from ' .. originalSize .. ')')

  return wordVocab, featuresVocabs
end

local function initVocabulary(name, dataFile, vocabFile, vocabSize, featuresVocabsFiles, maxSeqLength)
  local wordVocab
  local featuresVocabs = {}

  if vocabFile:len() > 0 then
    -- If given, load existing word dictionary.
    _G.logger:info('Reading ' .. name .. ' vocabulary from \'' .. vocabFile .. '\'...')
    wordVocab = onmt.utils.Dict.new()
    wordVocab:loadFile(vocabFile)
    _G.logger:info('Loaded ' .. wordVocab:size() .. ' ' .. name .. ' words')
  end

  if featuresVocabsFiles:len() > 0 then
    -- If given, discover existing features dictionaries.
    local j = 1

    while true do
      local file = featuresVocabsFiles .. '.' .. name .. '_feature_' .. j .. '.dict'

      if not path.exists(file) then
        break
      end

      _G.logger:info('Reading ' .. name .. ' feature ' .. j .. ' vocabulary from \'' .. file .. '\'...')
      featuresVocabs[j] = onmt.utils.Dict.new()
      featuresVocabs[j]:loadFile(file)
      _G.logger:info('Loaded ' .. featuresVocabs[j]:size() .. ' labels')

      j = j + 1
    end
  end

  if wordVocab == nil or (#featuresVocabs == 0 and hasFeatures(dataFile)) then
    -- If a dictionary is still missing, generate it.
    _G.logger:info('Building ' .. name  .. ' vocabulary...')
    local genWordVocab, genFeaturesVocabs = makeVocabulary(dataFile, vocabSize, maxSeqLength)

    if wordVocab == nil then
      wordVocab = genWordVocab
    end
    if #featuresVocabs == 0 then
      featuresVocabs = genFeaturesVocabs
    end
  end

  _G.logger:info('')

  return {
    words = wordVocab,
    features = featuresVocabs
  }
end

local function saveVocabulary(name, vocab, file)
  _G.logger:info('Saving ' .. name .. ' vocabulary to \'' .. file .. '\'...')
  vocab:writeFile(file)
end

local function saveFeaturesVocabularies(name, vocabs, prefix)
  for j = 1, #vocabs do
    local file = prefix .. '.' .. name .. '_feature_' .. j .. '.dict'
    _G.logger:info('Saving ' .. name .. ' feature ' .. j .. ' vocabulary to \'' .. file .. '\'...')
    vocabs[j]:writeFile(file)
  end
end

local function vecToTensor(vec)
  local t = torch.Tensor(#vec)
  for i, v in pairs(vec) do
    t[i] = v
  end
  return t
end

local function makeData(srcFile, tgtFile, srcDicts, tgtDicts)
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
        _G.logger:info('WARNING: source and target do not have the same number of sentences')
      end
      break
    end

    if isValid(srcTokens, opt.src_seq_length) and isValid(tgtTokens, opt.tgt_seq_length) then
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

    if count % opt.report_every == 0 then
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

  if opt.shuffle == 1 then
    _G.logger:info('... shuffling sentences')
    local perm = torch.randperm(#src)
    sizes = onmt.utils.Table.reorder(sizes, perm, true)
    reorderData(perm)
  end

  _G.logger:info('... sorting sentences by size')
  local _, perm = torch.sort(vecToTensor(sizes), true)
  reorderData(perm)

  _G.logger:info('Prepared ' .. #src .. ' sentences (' .. ignored
          .. ' ignored due to source length > ' .. opt.src_seq_length
          .. ' or target length > ' .. opt.tgt_seq_length .. ')')

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

local function main()
  local requiredOptions = {
    "train_src",
    "train_tgt",
    "valid_src",
    "valid_tgt",
    "save_data"
  }

  onmt.utils.Opt.init(opt, requiredOptions)

  _G.logger = onmt.utils.Logger.new(opt.log_file, true)

  local data = {}

  data.dicts = {}
  data.dicts.src = initVocabulary('source', opt.train_src, opt.src_vocab, opt.src_vocab_size,
                                  opt.features_vocabs_prefix, opt.src_seq_length)
  data.dicts.tgt = initVocabulary('target', opt.train_tgt, opt.tgt_vocab, opt.tgt_vocab_size,
                                  opt.features_vocabs_prefix, opt.tgt_seq_length)

  _G.logger:info('Preparing training data...')
  data.train = {}
  data.train.src, data.train.tgt = makeData(opt.train_src, opt.train_tgt,
                                            data.dicts.src, data.dicts.tgt)
  _G.logger:info('')

  _G.logger:info('Preparing validation data...')
  data.valid = {}
  data.valid.src, data.valid.tgt = makeData(opt.valid_src, opt.valid_tgt,
                                            data.dicts.src, data.dicts.tgt)
  _G.logger:info('')

  if opt.src_vocab:len() == 0 then
    saveVocabulary('source', data.dicts.src.words, opt.save_data .. '.src.dict')
  end

  if opt.tgt_vocab:len() == 0 then
    saveVocabulary('target', data.dicts.tgt.words, opt.save_data .. '.tgt.dict')
  end

  if opt.features_vocabs_prefix:len() == 0 then
    saveFeaturesVocabularies('source', data.dicts.src.features, opt.save_data)
    saveFeaturesVocabularies('target', data.dicts.tgt.features, opt.save_data)
  end

  _G.logger:info('Saving data to \'' .. opt.save_data .. '-train.t7\'...')
  torch.save(opt.save_data .. '-train.t7', data, 'binary', false)

end

main()
