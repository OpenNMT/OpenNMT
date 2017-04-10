require('onmt.init')

local tester = ...

local featuresTest = torch.TestSuite()

local function buildCaseFeatureDict()
  local dict = onmt.utils.Dict.new({onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                                    onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD})
  dict:add('C')
  dict:add('l')
  dict:add('n')
  dict:add('U')
  dict:add('M')
  return dict
end

function featuresTest.extract_empty()
  local tokens = {}
  local words, features, numFeatures = onmt.utils.Features.extract(tokens)
  tester:eq(words, tokens)
  tester:eq(features, {})
  tester:eq(numFeatures, 0)
end

function featuresTest.extract_none()
  local tokens = { 'Hello', 'World', '!' }
  local words, features, numFeatures = onmt.utils.Features.extract(tokens)
  tester:eq(words, tokens)
  tester:eq(features, {})
  tester:eq(numFeatures, 0)
end

function featuresTest.extract_unexpectedFeature()
  local tokens = { 'Hello', 'World￨unexpected', '!' }
  tester:assertError(function () onmt.utils.Features.extract(tokens) end)
end

function featuresTest.extract_nonEqualFeaturesCount()
  local tokens = { 'Hello￨C', 'World￨C', '!￨n￨PUNCT' }
  tester:assertError(function () onmt.utils.Features.extract(tokens) end)
end

function featuresTest.extract_one()
  local tokens = { 'Hello￨C', 'World￨C', '!￨n' }
  local words, features, numFeatures = onmt.utils.Features.extract(tokens)
  tester:eq(words, { 'Hello', 'World', '!' })
  tester:eq(features, { { 'C', 'C', 'n' } })
  tester:eq(numFeatures, 1)
end

function featuresTest.extract_two()
  local tokens = { 'Hello￨C￨INTERJ', 'World￨C￨NOUN', '!￨n￨PUNCT' }
  local words, features, numFeatures = onmt.utils.Features.extract(tokens)
  tester:eq(words, { 'Hello', 'World', '!' })
  tester:eq(features, { { 'C', 'C', 'n' }, { 'INTERJ', 'NOUN', 'PUNCT' } })
  tester:eq(numFeatures, 2)
end


function featuresTest.annotate_empty()
  local words = {}
  local features = {}
  local tokens = onmt.utils.Features.annotate(words, features)
  tester:eq(tokens, words)
end

function featuresTest.annotate_none()
  local words = { 'Hello', 'World', '!' }
  local features = {}
  local tokens = onmt.utils.Features.annotate(words, features)
  tester:eq(tokens, words)
end

function featuresTest.annotate_one()
  local words = { 'Hello', 'World', '!' }
  local features = { { 'C', 'C', 'n' } }
  local tokens = onmt.utils.Features.annotate(words, features)
  tester:eq(tokens, { 'Hello￨C', 'World￨C', '!￨n' })
end

function featuresTest.annotate_two()
  local words = { 'Hello', 'World', '!' }
  local features = { { 'C', 'C', 'n' }, { 'INTERJ', 'NOUN', 'PUNCT' } }
  local tokens = onmt.utils.Features.annotate(words, features)
  tester:eq(tokens, { 'Hello￨C￨INTERJ', 'World￨C￨NOUN', '!￨n￨PUNCT' })
end


local function generateSource(withTds)
  local features = { { 'C', 'C', 'n' } }
  local dicts = { buildCaseFeatureDict() }
  local ids = onmt.utils.Features.generateSource(dicts, features, withTds)
  if withTds then
    tester:eq(torch.typename(ids), 'tds.Vec')
  end
  tester:eq(#ids, 1)
  tester:eq(ids[1], torch.IntTensor({dicts[1]:lookup('C'), dicts[1]:lookup('C'), dicts[1]:lookup('n')}))
end


function featuresTest.generateSource_default()
  generateSource(tester)
end

function featuresTest.generateSource_defaultWithTds()
  generateSource(tester, true)
end


local function generateTarget(withTds, shifted)
  local features = { { 'C', 'C', 'n' } }
  local dicts = { buildCaseFeatureDict() }
  local ids = onmt.utils.Features.generateTarget(dicts, features, withTds, shifted)
  if withTds then
    tester:eq(torch.typename(ids), 'tds.Vec')
  end
  tester:eq(#ids, 1)
  local seq
  if shifted == 0 then
    seq = {onmt.Constants.BOS,
           dicts[1]:lookup('C'), dicts[1]:lookup('C'), dicts[1]:lookup('n'),
           onmt.Constants.EOS}
  else
    seq = {onmt.Constants.EOS, onmt.Constants.BOS,
           dicts[1]:lookup('C'), dicts[1]:lookup('C'), dicts[1]:lookup('n')}
  end
  tester:eq(ids[1], torch.IntTensor(seq))
end

function featuresTest.generateTarget_default()
  generateTarget()
end

function featuresTest.generateTarget_defaultWithTds()
  generateTarget(true)
end

function featuresTest.generateTarget_notShifted()
  generateTarget(false, 0)
end

function featuresTest.generateTarget_notShiftedWithTds()
  generateTarget(true, 0)
end

return featuresTest
