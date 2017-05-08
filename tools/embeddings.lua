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
      [[Path to outputted dict file from `preprocess.lua`.]],
      {
        valid = onmt.utils.ExtendedCmdLine.fileExists
      }
    },
    {
      '-embed_file', '',
      [[Path to the embedding file. Ignored if `-lang` is used.]],
      {
        valid = onmt.utils.ExtendedCmdLine.fileNullOrExists
      }
    },
    {
      '-save_data', '',
      [[Output file path/label.]],
      {
        valid = onmt.utils.ExtendedCmdLine.nonEmpty
      }
    },
    {
      '-save_unknown_dict', '',
      [[Path to file for saving vocabs not found in embedding.]]
    }
  }, 'Data')


cmd:setCmdLineOptions(
  {
    {
      '-lang', '',
      [[Wikipedia Language Code to autoload embeddings.]]
    },
    {
      '-embed_type', 'word2vec',
      [[Embeddings file origin. Ignored if `-lang` is used.]],
      {
        enum = {'word2vec', 'glove', 'fasttext'}
      }
    },
    {
      '-normalize', true,
      [[Boolean to normalize the word vectors, or not.]]
    },
    {
      '-approximate', false,
      [[If set, will also look for variants (case, joiner annotate) to match dictionary and word embedding.]]
    },
    {
      '-report_every', 100000,
      [[Print stats every this many lines read from embedding file.]],
      {
        valid = onmt.utils.ExtendedCmdLine.isInt(1)
      }
    }
  }, 'Embedding')

onmt.utils.Logger.declareOpts(cmd)

local opt = cmd:parse(arg)

--[[ Auto Loads language files from S3.

Embedding files made available by Rami Al-Rfou through Polygot (https://pypi.python.org/pypi/polyglot)]]
local function loadAuto(lang)
  local http = assert(require("socket.http"), 'autoload requires \'luasocket\' which can be installed via \'luarocks install luasocket\'')

  --TODO: Perhaps centralize?
  local endpoint = 'http://language-embedding-files.s3-website-us-east-1.amazonaws.com/'

  local filename = 'polyglot-' .. lang:lower() .. '.txt'
  local filepath = opt.save_data .. '-' .. filename

  if path.exists(filepath) then
    return filepath
  end

  local resp, stat = http.request(endpoint .. filename .. '.gz')

  if stat == 200 then
    local autoFile = io.open(filepath, 'w')
    local result = zlib.inflate()(resp)
    autoFile:write(result)
    autoFile:close()
  else
    error('embedding file for language code \'' .. lang .. '\' was not found')
  end

  return filepath
end

local function loadEmbeddings(embeddingFilename, embeddingType, dictionary)
  -- Converts binary to strings - Courtesy of https://github.com/rotmanmi/word2vec.torch
  local function readStringv2(file)
    local str = {}
    local maxW = 50

    for _ = 1, maxW do
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

  -- Looks for cased version and then variants of matching dictionary word.
  -- mark all variants so that if a exact version is found they are prefered
  local function locateIdxs(word, dict)
    local res = {}
    local idx = dict:lookup(word)

    if idx then
      res[idx] = true
    end

    if opt.approximate then
      -- looks for variants with joiner marks
      if word:find("￭") == nil then
        local res1 = locateIdxs("￭"..word, dict)
        for i in pairs(res1) do
          res[i] = false
        end
        local res2 = locateIdxs(word.."￭", dict)
        for i in pairs(res2) do
          res[i] = false
        end
        local res3 = locateIdxs("￭"..word.."￭", dict)
        for i in pairs(res3) do
          res[i] = false
        end
      end

      local lword = word:lower()
      idx = lword ~= word and dict:lookup(lword)
      if idx then
        res[idx] = false
      end
    end

    return res
  end


  -- Fills value for unmatched vocabs.
  local function fillGaps(weights, dict, loaded, dictSize, embeddingSize, save_unknown_dict)
    local fw
    local approximateCount = 0
    if save_unknown_dict ~= '' then
      fw = io.open(save_unknown_dict, 'w')
    end
    for idx = 1, dictSize do
      if loaded[idx] == nil then
        if fw then
          fw:write(dict:lookup(idx)..' '..idx..'\n')
        end
        for i = 1, embeddingSize do
          weights[idx][i] = torch.normal(0, 0.9)
        end
      elseif loaded[idx] == false then
        approximateCount = approximateCount + 1
      end
    end

    if fw then
      fw:close()
      _G.logger:info('* saved unknown vocabs in %s', save_unknown_dict)
    end

    return approximateCount, weights
  end

  -- Initializes OpenNMT constants.
  local function preloadSpecial (weights, loaded, dict, embeddingSize)
    local specials = {
      onmt.Constants.PAD_WORD,
      onmt.Constants.UNK_WORD,
      onmt.Constants.BOS_WORD,
      onmt.Constants.EOS_WORD
    }

    for i = 1, #specials do
      local res = locateIdxs(specials[i], dict)
      for idx, t in pairs(res) do
        -- replace previous approximation
        if loaded[idx] == nil or (loaded[idx] == false and t == true) then
          for e = 1, embeddingSize do
            weights[idx][e] = torch.normal(0, 0.9)
          end
          loaded[idx] = t
        end
      end
    end

    return weights, loaded

  end

  local function register(word, dict, loaded, weights, getWordEmbedding)
    local res = locateIdxs(word, dict)

    local wordEmbedding

    for idx, t in pairs(res) do
      -- replace previous approximation
      if loaded[idx] == nil or (loaded[idx] == false and t == true) then
        if not wordEmbedding then
          wordEmbedding = getWordEmbedding()
          -- Normalize word embedding.
          if opt.normalize then
          local norm = torch.norm(wordEmbedding, 2)
          if norm ~= 0 then
            wordEmbedding:div(norm)
          end
        end

        end
        weights[idx] = wordEmbedding
        loaded[idx] = t
      end
    end
  end

  -- Given a word2vec embedings file name and dictionary, outputs weights.
  -- Some portions are courtesy of https://github.com/rotmanmi/word2vec.torch
  local function loadWord2vec(f, dict)
    local loaded = tds.Hash()
    local dictSize = dict:size()

    -- Read header.
    f:ascii()
    local numWords = f:readInt()
    local embeddingSize = f:readInt()

    local weights = torch.Tensor(dictSize, embeddingSize)

    -- Preload constants.
    weights, loaded = preloadSpecial(weights, loaded, dict, embeddingSize)

    -- Read content.
    f:binary()

    for i = 1, numWords do
      if i % opt.report_every == 0 then
        _G.logger:info('... %d embeddings processed (%d/%d matched with the dictionary)',
                       i, #loaded, dictSize)
      end

      local word = readStringv2(f)
      local wordEmbedding = f:readFloat(embeddingSize)

      -- Skip newline.
      f:readChar()

      register(word, dict, loaded, weights, function()
        return torch.FloatTensor(wordEmbedding)
      end)

      -- End File loop
    end

    return weights, embeddingSize, loaded
  end

  -- Given a glove embedings file name and dictionary, outputs weights.
  local function loadGlove(f, dict)
    local loaded = tds.Hash()
    local dictSize = dict:size()
    local embeddingSize = nil
    local weights = nil
    local first = true
    local count = 0

    for line in f:lines() do
      count = count + 1
      if count % opt.report_every == 0 then
        _G.logger:info('... %d embeddings processed (%d/%d matched with the dictionary)',
                       count, #loaded, dictSize)
      end

      local splitLine = line:split(' ')

      if first == true then
        embeddingSize = #splitLine - 1
        weights = torch.Tensor(dictSize, embeddingSize)

        -- Preload constants.
        weights, loaded = preloadSpecial(weights, loaded, dict, embeddingSize)
        first = false
      end

      register(splitLine[1], dict, loaded, weights, function()
        local wordEmbedding = torch.Tensor(embeddingSize)
        for j = 2, #splitLine do
          wordEmbedding[j - 1] = tonumber(splitLine[j])
        end
        return wordEmbedding
      end)

      -- End File loop
    end

    return weights, embeddingSize, loaded
  end

  -- Given a glove embedings file name and dictionary, outputs weights.
  local function loadFasttext(f, dict)
    local loaded = tds.Hash()
    local dictSize = dict:size()
    local count = 0

    local header = f:read()
    local splitHeader = header:split(' ')
    assert(#splitHeader==2, "incorrect file format - header should be '#vocab dim'")
    local numWords = tonumber(splitHeader[1])
    local embeddingSize = tonumber(splitHeader[2])
    local weights = torch.Tensor(dictSize, embeddingSize)
    -- Preload constants.
    weights, loaded = preloadSpecial(weights, loaded, dict, embeddingSize)

    for line in f:lines() do
      count = count + 1
      if count % opt.report_every == 0 then
        _G.logger:info('... %d embeddings processed (%d/%d matched with the dictionary)',
                       count, #loaded, dictSize)
      end

      local splitLine = line:split(' ')

      register(splitLine[1], dict, loaded, weights, function()
        local wordEmbedding = torch.Tensor(embeddingSize)
        for j = 2, #splitLine do
          wordEmbedding[j - 1] = tonumber(splitLine[j])
        end
        return wordEmbedding
      end)

      -- End File loop
    end

    assert(count==numWords, "invalid line count")

    return weights, embeddingSize, loaded
  end

  _G.logger:info('Processing embedddings file \'%s\'...', embeddingFilename)

  local weights
  local embeddingSize
  local loaded

  local f = io.open(embeddingFilename, "r")
  if embeddingType == "word2vec" then
    weights, embeddingSize, loaded = loadWord2vec(f, dictionary)
  elseif embeddingType == "glove" then
    weights, embeddingSize, loaded = loadGlove(f, dictionary)
  elseif embeddingType == "fasttext" then
    weights, embeddingSize, loaded = loadFasttext(f, dictionary)
  end
  f:close()

  _G.logger:info('... done.')
  _G.logger:info(' * %d/%d embeddings matched with dictionary tokens', #loaded, dictionary:size())

  if #loaded ~= dictionary:size() then
    local approximateCount
    approximateCount, weights = fillGaps(weights, dictionary, loaded, dictionary:size(), embeddingSize, opt.save_unknown_dict)
    if approximateCount then
      _G.logger:info(' * %d approximate lookup', approximateCount)
    end
    _G.logger:info(' * %d/%d vocabs randomly assigned with a normal distribution', dictionary:size() - #loaded, dictionary:size())
  end

  return weights, embeddingSize
end


local function main()
  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)

  local timer = torch.Timer()

  local dict = onmt.utils.Dict.new(opt.dict_file)

  local embedFile = opt.embed_file
  local embedType = opt.embed_type

  if opt.lang and opt.lang:len() > 0 then
    _G.logger:info('Running autoload for %s...', opt.lang)
    embedFile = loadAuto(opt.lang)
    embedType = 'glove'
  end

  local weights, embeddingSize = loadEmbeddings(embedFile, embedType, dict)

  local targetFile = opt.save_data .. '-embeddings-' .. tostring(embeddingSize) .. '.t7'
  _G.logger:info('Saving embeddings to \'%s\'...', targetFile)
  torch.save(targetFile, weights)

  _G.logger:info('Completed in %0.3f seconds. ', timer:time().real)
  _G.logger:info('')
  _G.logger:info('For source embeddings, set the options \'-pre_word_vecs_enc %s -src_word_vec_size %d\' with train.lua', targetFile, embeddingSize)
  _G.logger:info('For target embeddings, set the options \'-pre_word_vecs_dec %s -tgt_word_vec_size %d\' with train.lua', targetFile, embeddingSize)
end

main()
