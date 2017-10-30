require('onmt.init')

local path = require('pl.path')

local tester = ...

local vocabularyTest = torch.TestSuite()

local dataDir = 'data'
local testDataDir = 'test/data'
local noFilter = function (_) return true end
local filterShortSentences = function(sent) return #sent>10 end

function vocabularyTest.filterAll()
  local filterAll = function (t) return #t == 0 end
  local wordVocab, featuresVocabs = onmt.data.Vocabulary.make(dataDir .. '/src-val.txt', filterAll)

  tester:eq(wordVocab:size(), 4)
  tester:eq(#featuresVocabs, 0)
end

function vocabularyTest.initSimple()
  local vocabs = onmt.data.Vocabulary.init('source', dataDir .. '/src-val.txt', '', { 1000 }, { 0 }, '', noFilter)

  tester:eq(vocabs.words:size(), 1004)
  tester:eq(#vocabs.features, 0)
  tester:eq(#vocabs.words.special, 4)

  onmt.data.Vocabulary.save('source', vocabs.words, 'src.dict')
  tester:ne(path.exists('src.dict'), false)

  local reused = onmt.data.Vocabulary.init('source', dataDir .. '/src-val.txt', 'src.dict', { 50000 }, '0', '', noFilter)

  tester:eq(vocabs.words:size(), reused.words:size())

  os.remove('src.dict')
end

function vocabularyTest.keepFrequency()
  local vocabs = onmt.data.Vocabulary.init('source', dataDir .. '/src-val.txt', '', { 1000 }, { 0 }, '', noFilter, true)

  tester:eq(vocabs.words:size(), 1004)
  tester:eq(vocabs.words.freqTensor:dim(), 1)
  tester:eq(vocabs.words.freqTensor:size(1), 1004)
  -- checking unknown word frequency
  tester:eq(vocabs.words.freqTensor[2],19699)
  tester:eq(vocabs.words.freqTensor[10],1445)

  -- check also that frequency is recalculated when using saved dictionaries
  vocabs = onmt.data.Vocabulary.init('source', dataDir .. '/src-val.txt', '', { 1000 }, { 0 }, '', noFilter, false)
  tester:eq(vocabs.words.freqTensor, nil)
  onmt.data.Vocabulary.save('source', vocabs.words, 'src.dict')
  local reused = onmt.data.Vocabulary.init('source', dataDir .. '/src-val.txt', 'src.dict', { 1000 }, { 0 }, '', noFilter, true)

  tester:ne(reused.words.freqTensor, nil)
  tester:eq(reused.words.freqTensor[2],19699)
  tester:eq(reused.words.freqTensor[10],1445)

  os.remove('src.dict')

end

function vocabularyTest.minFrequency()
  local vocabs = onmt.data.Vocabulary.init('source', dataDir .. '/src-val.txt', '', { 0 }, { 5 }, '', noFilter, true)

  tester:eq(vocabs.words:size(), 1957)
end

function vocabularyTest.filterSent()
  local vocabs = onmt.data.Vocabulary.init('source', dataDir .. '/src-val.txt', '', { 0 }, { 5 }, '', filterShortSentences)

  tester:eq(vocabs.words:size(), 1864)
end

function vocabularyTest.placeholder()
  local vocabs = onmt.data.Vocabulary.init('source', testDataDir .. '/text-placeholder.tok', '', { 1000 }, { 1 }, '', noFilter, true)
  tester:assert(vocabs.words:lookup("｟ent_url＃1｠") ~= nil)
end

function vocabularyTest.initFeatures()
  local vocabs = onmt.data.Vocabulary.init('source', dataDir .. '/src-val-case.txt', '', { 1000, 4 }, { 0 }, '', noFilter)

  tester:eq(#vocabs.features, 1)
  tester:eq(vocabs.features[1]:size(), 8)

  onmt.data.Vocabulary.save('source', vocabs.words, 'src.dict')
  onmt.data.Vocabulary.saveFeatures('source', vocabs.features, 'test.source')
  tester:ne(path.exists('test.source_feature_1.dict'), false)

  local reuseFeatOnly = onmt.data.Vocabulary.init('source', dataDir .. '/src-val-case.txt', '', { 2000 }, { 0 }, 'test', noFilter)

  tester:eq(reuseFeatOnly.words:size(), 2004)
  tester:eq(reuseFeatOnly.features[1]:size(), vocabs.features[1]:size())

  local reuseBoth = onmt.data.Vocabulary.init('source', dataDir .. '/src-val-case.txt', 'src.dict', { 2000 }, { 0 }, 'test', noFilter)

  tester:eq(reuseBoth.words:size(), vocabs.words:size())
  tester:eq(reuseBoth.features[1]:size(), vocabs.features[1]:size())

  os.remove('src.dict')
  os.remove('test.source_feature_1.dict')
end

return vocabularyTest
