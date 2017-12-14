require('onmt.init')

local tester = ...

local preprocessorTest = torch.TestSuite()
local hookManager_save

local dataDir = 'data'

local noFilter = function(_) return true end

local function buildPreprocessor(mode)
  local cmd = onmt.utils.ExtendedCmdLine.new()
  onmt.data.Preprocessor.declareOpts(cmd, mode == 'parsedir' and 'bitext' or mode)
  onmt.data.Preprocessor.expandOpts(cmd, mode == 'parsedir' and 'bitext' or mode)

  local commandLine
  if not mode or mode == 'bitext' then
    commandLine = {
      '-train_src', dataDir .. '/src-val.txt',
      '-train_tgt', dataDir .. '/tgt-val.txt',
      '-valid_src', dataDir .. '/src-test.txt',
      '-valid_tgt', dataDir .. '/tgt-test.txt'
    }
  elseif mode == 'monotext' then
    commandLine = {
      '-train', dataDir .. '/src-val-case.txt',
      '-valid', dataDir .. '/src-test-case.txt'
    }
  elseif mode == 'feattext' then
    commandLine = {
      '-train_src', dataDir .. '/sigtraintrig.srcfeat',
      '-train_tgt', dataDir .. '/sigtraintrig.tgt',
      '-valid_src', dataDir .. '/sigvaltrig.srcfeat',
      '-valid_tgt', dataDir .. '/sigvaltrig.tgt',
      '-idx_files',
      '-src_seq_length', 100
    }
  elseif mode == 'parsedir' then
    commandLine = {
      '-train_dir', dataDir,
      '-src_suffix', 'src-val.txt',
      '-tgt_suffix', 'tgt-val.txt',
      '-valid_src', dataDir .. '/src-test.txt',
      '-valid_tgt', dataDir .. '/tgt-test.txt',
      '-src_vocab', 'ddict.src.dict',
      '-tgt_vocab', 'ddict.tgt.dict',
      '-tok_src_mode', 'conservative',
      '-gsample', 0.1,
      '-gsample_dist', 'test/data/drule',
      '-src_seq_length', 10,
      '-preprocess_pthreads', 1
    }
  end

  local opt = cmd:parse(commandLine)

  local preprocessor = onmt.data.Preprocessor.new(opt, mode == 'parsedir' and 'bitext' or mode)

  return preprocessor, opt
end

local function makeDicts(srctgt, file)
  return onmt.data.Vocabulary.init(srctgt, file, '',  { 0 }, { 0 }, '', noFilter)
end

function preprocessorTest.bitext()
  hookManager_save = _G.hookManager

  local preprocessor, opt = buildPreprocessor("bitext")

  local srcDicts = makeDicts('source',opt.train_src)
  local tgtDicts = makeDicts('target',opt.train_tgt)

  local srcData, tgtData = preprocessor:makeBilingualData({{1,{opt.train_src, opt.train_tgt}}}, srcDicts, tgtDicts)

  tester:eq(torch.typename(srcData.words), 'tds.Vec')
  tester:eq(torch.typename(srcData.features), 'tds.Vec')
  tester:eq(#srcData.words, 2819)

  tester:eq(torch.typename(tgtData.words), 'tds.Vec')
  tester:eq(torch.typename(tgtData.features), 'tds.Vec')
  tester:eq(#tgtData.words, 2819)

  onmt.data.Vocabulary.save('source', srcDicts.words, 'ddict.src.dict')
  onmt.data.Vocabulary.save('target', tgtDicts.words, 'ddict.tgt.dict')

  preprocessor = buildPreprocessor('parsedir')
  local dicts = preprocessor:getVocabulary()
  local data = preprocessor:makeData('train', dicts)
  -- sample 10% and sentence length<=10
  tester:assertle(#data.src.words, 60)

  os.remove('ddict.src.dict')
  os.remove('ddict.tgt.dict')

  _G.hookManager = hookManager_save
end

function preprocessorTest.monotext()
  hookManager_save = _G.hookManager

  local preprocessor, opt = buildPreprocessor('monotext')

  local dicts = makeDicts('source',opt.train)

  local data = preprocessor:makeMonolingualData({{1,{opt.train}}}, dicts)

  tester:eq(torch.typename(data.words), 'tds.Vec')
  tester:eq(torch.typename(data.features), 'tds.Vec')
  tester:eq(#data.words, 2857)
  tester:eq(#data.features, 2857)

  _G.hookManager = hookManager_save
end

function preprocessorTest.feattext()
  hookManager_save = _G.hookManager

  local preprocessor, opt = buildPreprocessor('feattext')

  local tgtDicts = makeDicts('target',opt.train_tgt)

  local srcData,tgtData = preprocessor:makeFeatTextData({{1,{opt.train_src, opt.train_tgt}}}, tgtDicts)

  tester:eq(torch.typename(srcData.vectors), 'tds.Vec')
  tester:eq(torch.typename(tgtData.words), 'tds.Vec')
  tester:eq(torch.typename(tgtData.features), 'tds.Vec')
  tester:eq(srcData.vectors[1]:size(2), 2)
  tester:eq(#srcData.vectors, 947)
  tester:eq(#tgtData.features, 0)

  _G.hookManager = hookManager_save
end

return preprocessorTest
