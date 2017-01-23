local path = require('pl.path')

--[[ Vocabulary management utility functions. ]]
local Vocabulary = torch.class("Vocabulary")

local function hasFeatures(filename)
  local reader = onmt.utils.FileReader.new(filename)
  local _, _, numFeatures = onmt.utils.Features.extract(reader:next())
  reader:close()
  return numFeatures > 0
end

function Vocabulary.make(filename, size, validFunc)
  local wordVocab = onmt.utils.Dict.new({onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                                         onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD})
  local featuresVocabs = {}

  local reader = onmt.utils.FileReader.new(filename)

  while true do
    local sent = reader:next()
    if sent == nil then
      break
    end

    if validFunc(sent) then
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

function Vocabulary.init(name, dataFile, vocabFile, vocabSize, featuresVocabsFiles, validFunc)
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
    local genWordVocab, genFeaturesVocabs = Vocabulary.make(dataFile, vocabSize, validFunc)

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

function Vocabulary.save(name, vocab, file)
  _G.logger:info('Saving ' .. name .. ' vocabulary to \'' .. file .. '\'...')
  vocab:writeFile(file)
end

function Vocabulary.saveFeatures(name, vocabs, prefix)
  for j = 1, #vocabs do
    local file = prefix .. '.' .. name .. '_feature_' .. j .. '.dict'
    _G.logger:info('Saving ' .. name .. ' feature ' .. j .. ' vocabulary to \'' .. file .. '\'...')
    vocabs[j]:writeFile(file)
  end
end

return Vocabulary
