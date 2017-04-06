require('torch')
require('onmt.init')

local tds = require('tds')
local zlib = require ('zlib')
local path = require('pl.path')

local cmd = onmt.utils.ExtendedCmdLine.new('embeddings.lua')

cmd:setCmdLineOptions(
  {
    {
      '-dict_file', '',
      [[Path to outputted dict file from preprocess.lua.]]
    },
    {
      '-embed_file', '',
      [[Path to embedding file. Ignored if auto_lang is used.]]
    },
    {
      '-save_data', '',
      [[Output file path/label.]]
    }
  }, 'Data')


cmd:setCmdLineOptions(
  {
    {
      '-auto_lang', '',
      [[Wikipedia Language Code to autoload embeddings.]]
    },
    {
      '-embed_type', 'word2vec',
      [['word2vec' or 'glove'. Ignored if auto_lang is used.]]
    },
    {
      '-normalize', 'true',
      [[Boolean to normalize the word vectors, or not.]]
    },
    {
      '-report_every', '100000',
      [[Print stats every this many lines read from embedding file.]]
    }
  }, 'Embedding')

local opt = cmd:parse(arg)

-- [[Auto Loads language files from S3]]
-- [[Embedding files made available by Rami Al-Rfou through Polygot ( Project: https://pypi.python.org/pypi/polyglot Paper: http://www.aclweb.org/anthology/W13-3520 )]]
local function loadAuto(lang)

  local http = assert(require("socket.http"), 'autoload requires \'luasocket\' which can be installed via \'luarocks install luasocket\'')

  --TODO: Perhaps centralize?
  local endpoint = 'http://language-embedding-files.s3-website-us-east-1.amazonaws.com/'


  local filename = 'polyglot-' .. lang:lower() .. '.txt'
  local filepath = opt.save_data .. "-" .. filename

  if path.exists(filepath) then
    return filepath
  end

  local resp, stat = http.request(endpoint .. filename .. '.gz')

  if stat == 200 then

    local autoFile = io.open(filepath, "w")

    local result = zlib.inflate()(resp)

    autoFile:write(result)
    autoFile:close()

  else

    error('embedding file for language code \'' .. lang .. '\' was not found')

  end

  return filepath

end



local function loadEmbeddings(embeddingFilename, embeddingType, dictionary)


  --[[Converts binary to strings - Courtesy of https://github.com/rotmanmi/word2vec.torch]]
  local function readStringv2(file)
    local str = {}
    local max_w = 50

    for _ = 1, max_w do
      local char = file:readChar()

      if char == 32 or char == 10 or char == 0 then
        break
      else
        str[#str + 1] = char
      end
    end

    str = torch.CharStorage(str)
    return str:string()

  end

  -- [[Looks for cased version and then lower version of matching dictionary word.]]
  local function locateIdx(word, dict)

    local idx = nil

    if dict:lookup(word) ~= nil then
      idx = dict:lookup(word)

    elseif dict:lookup(word:lower()) ~= nil then
      idx = dict:lookup(word:lower())

    end

    return idx

  end


  -- [[Fills value for unmatched embeddings]]
  local function fillGaps(weights, loaded, dictSize, embeddingSize)

    for idx = 1, dictSize do
      if loaded[idx] == nil then
        for i=1, embeddingSize do
          weights[idx][i] = torch.uniform(-1, 1)
        end
      end
    end

    return weights

  end

  -- [[Initializes OpenNMT constants.]]
  local function preloadSpecial (weights, loaded, dict, embeddingSize)

    local specials = {onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD, onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD}

    for i = 1, #specials do
      local idx = locateIdx(specials[i], dict)
      for e=1, embeddingSize do
        weights[idx][e] = torch.normal(0, 0.9)
      end
      loaded[idx] = true
    end

    return weights, loaded

  end

  --[[Given a word2vec embedings file name and dictionary, outputs weights. Some portions courtesy of Courtesy of https://github.com/rotmanmi/word2vec.torch]]
  local function loadWord2vec(filename, dict)

    local loaded = tds.Hash()
    local dictSize = dict:size()

    local f = torch.DiskFile(filename, "r")

    -- read header
    f:ascii()
    local numWords = f:readInt()
    local embeddingSize = f:readInt()


    local weights = torch.Tensor(dictSize, embeddingSize)

    -- preload constants
    weights, loaded = preloadSpecial (weights, loaded, dict, embeddingSize)

    -- read content
    f:binary()

    print('processing embeddding file')
    for i = 1, numWords do

      if i%opt.report_every == 0 then
         print(i .. ' embedding tokens reviewed. ' .. #loaded .. ' out of ' .. dictSize .. ' dictionary tokens matched.' )
      end

      local word = readStringv2(f)
      local wordEmbedding = f:readFloat(embeddingSize)
      wordEmbedding = torch.FloatTensor(wordEmbedding)

      local idx = locateIdx(word, dict)

      if idx ~= nil then

        local norm = torch.norm(wordEmbedding, 2)

        -- normalize word embedding
        if norm ~= 0 and opt.normalize == true then
          wordEmbedding:div(norm)
        end

        weights[idx] = wordEmbedding
        loaded[idx] = true

      end

      if #loaded == dictSize then
        print('Quitting early. All ' .. dictSize .. ' dictionary tokens matched.')
        break
      end

    -- End File loop
    end

    if #loaded ~= dictSize then
      print('Embedding file fully processed. ' .. #loaded .. ' out of ' .. dictSize .. ' dictionary tokens matched.')
      weights = fillGaps(weights, loaded, dictSize, embeddingSize)
      print('Remaining randomly assigned according to uniform distribution')
    end

    return weights, embeddingSize

  end


  --[[Given a glove embedings file name and dictionary, outputs weights ]]
  local function loadGlove(filename, dict)
    local loaded = tds.Hash()
    local dictSize = dict:size()
    local embeddingSize = nil
    local weights = nil
    local first = true
    local count = 0

    local f = io.open(filename, "r")

    print('processing embeddding file')
    for line in f:lines() do

      count = count + 1
      if count%opt.report_every == 0 then
         print(count .. ' embedding tokens reviewed. ' .. #loaded .. ' out of ' .. dictSize .. ' dictionary tokens matched.' )
      end

      local splitLine = line:split(' ')

      if first == true then
        embeddingSize = #splitLine - 1
        weights = torch.Tensor(dictSize, embeddingSize)

        -- preload constants
        weights, loaded = preloadSpecial (weights, loaded, dict, embeddingSize)
        first = false
      end

      local word = splitLine[1]
      local idx = locateIdx(word, dict)

      if idx ~= nil then

        local wordEmbedding = torch.Tensor(embeddingSize)

        for j = 2, #splitLine do
          wordEmbedding[j - 1] = tonumber(splitLine[j])
        end

        local norm = torch.norm(wordEmbedding, 2)

        -- normalize word embedding
        if norm ~= 0 and opt.normalize == true then
          wordEmbedding:div(norm)
        end

        weights[idx] = wordEmbedding
        loaded[idx] = true

      end

      if #loaded == dictSize then
        print('Quitting early. All ' .. dictSize .. ' dictionary tokens matched.')
        break
      end

    -- End File loop
    end

	f:close()

    if #loaded ~= dictSize then
      print('Embedding file fully processed. ' .. #loaded .. ' out of ' .. dictSize .. ' dictionary tokens matched.')
      weights = fillGaps(weights, loaded, dictSize, embeddingSize)
      print('Remaining randomly assigned according to uniform distribution')
    end

    return weights, embeddingSize

  end



  if embeddingType == "word2vec" then

    return loadWord2vec(embeddingFilename, dictionary)

  elseif embeddingType == "glove" then

    return loadGlove(embeddingFilename, dictionary)

  else

    error('invalid embed type. \'word2vec\' and \'glove\' are the only options.')

  end

end



local function main()

  local timer = torch.Timer()

  assert(path.exists(opt.dict_file), 'dictionary file \'' .. opt.dict_file .. '\' does not exist.')

  local dict = onmt.utils.Dict.new(opt.dict_file)

  local embedFile = opt.embed_file
  local embedType = opt.embed_type

  if opt.auto_lang and opt.auto_lang:len() > 0 then

    print('running autoload for ' .. opt.auto_lang)
    embedFile = loadAuto(opt.auto_lang)
    embedType = 'glove'

  end

  assert(path.exists(embedFile), 'embeddings file \'' .. opt.embed_file .. '\' does not exist.')
  local weights, embeddingSize = loadEmbeddings(embedFile, embedType, dict)

  print('saving weights: ' .. opt.save_data .. '-embeddings-' .. tostring(embeddingSize) .. '.t7' )
  torch.save(opt.save_data .. '-embeddings-' .. tostring(embeddingSize) .. '.t7', weights)

  print(string.format('completed in %0.3f seconds. ',timer:time().real) .. ' embedding vector size is: ' .. embeddingSize )


end

main()
