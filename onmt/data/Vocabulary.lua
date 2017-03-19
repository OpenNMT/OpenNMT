local path = require('pl.path')

--[[ Vocabulary management utility functions. ]]
local Vocabulary = torch.class("Vocabulary")

local function countFeatures(filename, idxFile)
  local reader = onmt.utils.FileReader.new(filename, idxFile)
  local _, _, numFeatures = onmt.utils.Features.extract(reader:next())
  reader:close()
  return numFeatures
end

function Vocabulary.make(filename, validFunc, idxFile)
  local wordVocab = onmt.utils.Dict.new({onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                                         onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD})
  local featuresVocabs = {}

  local reader = onmt.utils.FileReader.new(filename, idxFile)
  local lineId = 0

  while true do
    local sent = reader:next()
    if sent == nil then
      break
    end

    lineId = lineId + 1

    if validFunc(sent) then
      local words, features, numFeatures
      local _, err = pcall(function ()
        words, features, numFeatures = onmt.utils.Features.extract(sent)
      end)

      if err then
        error(err .. ' (' .. filename .. ':' .. lineId .. ')')
      end

      if #featuresVocabs == 0 and numFeatures > 0 then
        for j = 1, numFeatures do
          featuresVocabs[j] = onmt.utils.Dict.new({onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                                                   onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD})
        end
      else
        assert(#featuresVocabs == numFeatures,
               'all sentences must have the same numbers of additional features (' .. filename .. ':' .. lineId .. ')')
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

  return wordVocab, featuresVocabs
end

function Vocabulary.init(name, dataFile, vocabFile, vocabSize, wordsMinFrequency, featuresVocabsFiles, validFunc, idxFile)
  local wordVocab
  local featuresVocabs = {}
  local numFeatures = countFeatures(dataFile, idxFile)

  if vocabFile:len() > 0 then
    -- If given, load existing word dictionary.
    _G.logger:info('Reading ' .. name .. ' vocabulary from \'' .. vocabFile .. '\'...')
    wordVocab = onmt.utils.Dict.new()
    wordVocab:loadFile(vocabFile)
    _G.logger:info('Loaded ' .. wordVocab:size() .. ' ' .. name .. ' words')
  end

  if featuresVocabsFiles:len() > 0 and numFeatures > 0 then
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

    assert(#featuresVocabs > 0,
           'dictionary \'' .. featuresVocabsFiles .. '.' .. name .. '_feature_1.dict\' not found')
    assert(#featuresVocabs == numFeatures,
           'the data contains ' .. numFeatures .. ' ' .. name
             .. ' features but only ' .. #featuresVocabs .. ' dictionaries were found')
  end

  if wordVocab == nil or (#featuresVocabs == 0 and numFeatures > 0) then
    -- If a dictionary is still missing, generate it.
    _G.logger:info('Building ' .. name  .. ' vocabularies...')
    local genWordVocab, genFeaturesVocabs = Vocabulary.make(dataFile, validFunc, idxFile)

    local originalSizes = { genWordVocab:size() }
    for i = 1, #genFeaturesVocabs do
      table.insert(originalSizes, genFeaturesVocabs[i]:size())
    end

    local newSizes = onmt.utils.String.split(vocabSize, ',')
    local minFrequency = onmt.utils.String.split(wordsMinFrequency, ',')

    for i = 1, 1 + #genFeaturesVocabs do
      newSizes[i] = (newSizes[i] and tonumber(newSizes[i])) or 0
      minFrequency[i] = (minFrequency[i] and tonumber(minFrequency[i])) or 0
    end

    if wordVocab == nil then
      if minFrequency[1] > 0 then
        wordVocab = genWordVocab:pruneByMinFrequency(minFrequency[1])
      elseif newSizes[1] > 0 then
        wordVocab = genWordVocab:prune(newSizes[1])
      else
        wordVocab = genWordVocab
      end

      _G.logger:info('Created word dictionary of size '
                       .. wordVocab:size() .. ' (pruned from ' .. originalSizes[1] .. ')')
    end

    if #featuresVocabs == 0 then
      for i = 1, #genFeaturesVocabs do
        if minFrequency[i + 1] > 0 then
          featuresVocabs[i] = genFeaturesVocabs[i]:pruneByMinFrequency(minFrequency[i + 1])
        elseif newSizes[i + 1] > 0 then
          featuresVocabs[i] = genFeaturesVocabs[i]:prune(newSizes[i + 1])
        else
          featuresVocabs[i] = genFeaturesVocabs[i]
        end

        _G.logger:info('Created feature ' .. i .. ' dictionary of size '
                         .. featuresVocabs[i]:size() .. ' (pruned from ' .. originalSizes[i + 1] .. ')')

      end
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
