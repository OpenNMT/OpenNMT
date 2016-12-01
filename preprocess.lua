require('./lib/utils')

local constants = require('lib.constants')
local path = require('pl.path')

local cmd = torch.CmdLine()

cmd:text("")
cmd:text("preprocess.lua")
cmd:text("")
cmd:text("**Preprocess Options**")
cmd:text("")
cmd:text("")
cmd:option('-config', '', [[Read options from this file]])

cmd:option('-train_src_file', '', [[Path to the training source data]])
cmd:option('-train_targ_file', '', [[Path to the training target data]])
cmd:option('-valid_src_file', '', [[Path to the validation source data]])
cmd:option('-valid_targ_file', '', [[Path to the validation target data]])

cmd:option('-output_file', '', [[Output file for the prepared data]])

cmd:option('-src_vocab_size', 50000, [[Size of the source vocabulary]])
cmd:option('-targ_vocab_size', 50000, [[Size of the target vocabulary]])
cmd:option('-src_vocab_file', '', [[Path to an existing source vocabulary]])
cmd:option('-targ_vocab_file', '', [[Path to an existing target vocabulary]])
cmd:option('-features_vocabs_prefix', '', [[Path prefix to existing features vocabularies]])

cmd:option('-seq_length', 50, [[Maximum sequence length]])
cmd:option('-shuffle', 1, [[Suffle data]])
cmd:option('-seed', 3435, [[Random seed]])

cmd:option('-report_every', 100000, [[Report status every this many sentences]])

local opt = cmd:parse(arg)

local function hasFeatures(filename)
  local reader = utils.FileReader.new(filename)
  local _, _, num_features = utils.Features.extract(reader:next())
  reader:close()
  return num_features > 0
end

local function make_vocabulary(filename, size)
  local word_vocab = utils.Dict.new({constants.PAD_WORD, constants.UNK_WORD,
                                     constants.BOS_WORD, constants.EOS_WORD})
  local features_vocabs = {}

  local reader = utils.FileReader.new(filename)

  while true do
    local sent = reader:next()
    if sent == nil then
      break
    end

    local words, features, num_features = utils.Features.extract(sent)

    if #features_vocabs == 0 and num_features > 0 then
      for j = 1, num_features do
        features_vocabs[j] = utils.Dict.new({constants.PAD_WORD, constants.UNK_WORD,
                                             constants.BOS_WORD, constants.EOS_WORD})
      end
    else
      assert(#features_vocabs == num_features,
             'all sentences must have the same numbers of additional features')
    end

    for i = 1, #words do
      word_vocab:add(words[i])

      for j = 1, num_features do
        features_vocabs[j]:add(features[j][i])
      end
    end

  end

  reader:close()

  local original_size = #word_vocab
  word_vocab = word_vocab:prune(size)
  print('Created dictionary of size ' .. #word_vocab .. ' (pruned from ' .. original_size .. ')')

  return word_vocab, features_vocabs
end

local function init_vocabulary(name, data_file, vocab_file, vocab_size, features_vocabs_files)
  local word_vocab
  local features_vocabs = {}

  if vocab_file:len() > 0 then
    -- If given, load existing word dictionary.
    print('Reading ' .. name .. ' vocabulary from \'' .. vocab_file .. '\'...')
    word_vocab = utils.Dict.new()
    word_vocab:load_file(vocab_file)
    print('Loaded ' .. #word_vocab .. ' ' .. name .. ' words')
  end

  if features_vocabs_files:len() > 0 then
    -- If given, discover existing features dictionaries.
    local j = 1

    while true do
      local file = features_vocabs_files .. '.' .. name .. '_feature_' .. j .. '.dict'

      if not path.exists(file) then
        break
      end

      print('Reading ' .. name .. ' feature ' .. j .. ' vocabulary from \'' .. file .. '\'...')
      features_vocabs[j] = utils.Dict.new()
      features_vocabs[j]:load_file(file)
      print('Loaded ' .. #features_vocabs[j] .. ' labels')

      j = j + 1
    end
  end

  if word_vocab == nil or (#features_vocabs == 0 and hasFeatures(data_file)) then
    -- If a dictionary is still missing, generate it.
    print('Building ' .. name  .. ' vocabulary...')
    local gen_word_vocab, gen_features_vocabs = make_vocabulary(data_file, vocab_size)

    if word_vocab == nil then
      word_vocab = gen_word_vocab
    end
    if #features_vocabs == 0 then
      features_vocabs = gen_features_vocabs
    end
  end

  print('')

  return {
    words = word_vocab,
    features = features_vocabs
  }
end

local function save_vocabulary(name, vocab, file)
  print('Saving ' .. name .. ' vocabulary to \'' .. file .. '\'...')
  vocab:write_file(file)
end

local function save_features_vocabularies(name, vocabs, prefix)
  for j = 1, #vocabs do
    local file = prefix .. '.' .. name .. '_feature_' .. j .. '.dict'
    print('Saving ' .. name .. ' feature ' .. j .. ' vocabulary to \'' .. file .. '\'...')
    vocabs[j]:write_file(file)
  end
end

local function make_data(src_file, targ_file, src_dicts, targ_dicts)
  local src = {}
  local src_features = {}

  local targ = {}
  local targ_features = {}

  local sizes = {}

  local count = 0
  local ignored = 0

  local src_reader = utils.FileReader.new(src_file)
  local targ_reader = utils.FileReader.new(targ_file)

  while true do
    local src_tokens = src_reader:next()
    local targ_tokens = targ_reader:next()

    if src_tokens == nil or targ_tokens == nil then
      if src_tokens == nil and targ_tokens ~= nil or src_tokens ~= nil and targ_tokens == nil then
        print('WARNING: source and target do not have the same number of sentences')
      end
      break
    end

    if #src_tokens > 0 and #src_tokens <= opt.seq_length
    and #targ_tokens > 0 and #targ_tokens <= opt.seq_length then
      local src_words, src_feats = utils.Features.extract(src_tokens)
      local targ_words, targ_feats = utils.Features.extract(targ_tokens)

      table.insert(src, src_dicts.words:convert_to_idx(src_words, constants.UNK_WORD))
      table.insert(targ, targ_dicts.words:convert_to_idx(targ_words, constants.UNK_WORD,
                                                         constants.BOS_WORD, constants.EOS_WORD))

      if #src_dicts.features > 0 then
        table.insert(src_features, utils.Features.generateSource(src_dicts.features, src_feats))
      end
      if #targ_dicts.features > 0 then
        table.insert(targ_features, utils.Features.generateTarget(targ_dicts.features, targ_feats))
      end

      table.insert(sizes, #src_words)
    else
      ignored = ignored + 1
    end

    count = count + 1

    if count % opt.report_every == 0 then
      print('... ' .. count .. ' sentences prepared')
    end
  end

  src_reader:close()
  targ_reader:close()

  if opt.shuffle == 1 then
    print('... shuffling sentences')
    local perm = torch.randperm(#src)
    src = utils.Table.reorder(src, perm)
    targ = utils.Table.reorder(targ, perm)
    sizes = utils.Table.reorder(sizes, perm)

    if #src_dicts.features > 0 then
      src_features = utils.Table.reorder(src_features, perm)
    end
    if #targ_dicts.features > 0 then
      targ_features = utils.Table.reorder(targ_features, perm)
    end
  end

  print('... sorting sentences by size')
  local _, perm = torch.sort(torch.Tensor(sizes))
  src = utils.Table.reorder(src, perm)
  targ = utils.Table.reorder(targ, perm)

  if #src_dicts.features > 0 then
    src_features = utils.Table.reorder(src_features, perm)
  end
  if #targ_dicts.features > 0 then
    targ_features = utils.Table.reorder(targ_features, perm)
  end

  print('Prepared ' .. #src .. ' sentences (' .. ignored .. ' ignored due to length == 0 or > ' .. opt.seq_length .. ')')

  local src_data = {
    words = src,
    features = src_features
  }

  local targ_data = {
    words = targ,
    features = targ_features
  }

  return src_data, targ_data
end

local function main()
  local required_options = {
    "train_src_file",
    "train_targ_file",
    "valid_src_file",
    "valid_targ_file",
    "output_file"
  }

  utils.Opt.init(opt, required_options)

  local data = {}

  data.dicts = {}
  data.dicts.src = init_vocabulary('source', opt.train_src_file, opt.src_vocab_file,
                                   opt.src_vocab_size, opt.features_vocabs_prefix)
  data.dicts.targ = init_vocabulary('target', opt.train_targ_file, opt.targ_vocab_file,
                                    opt.targ_vocab_size, opt.features_vocabs_prefix)

  print('Preparing training data...')
  data.train = {}
  data.train.src, data.train.targ = make_data(opt.train_src_file, opt.train_targ_file,
                                              data.dicts.src, data.dicts.targ)
  print('')

  print('Preparing validation data...')
  data.valid = {}
  data.valid.src, data.valid.targ = make_data(opt.valid_src_file, opt.valid_targ_file,
                                              data.dicts.src, data.dicts.targ)
  print('')

  if opt.src_vocab_file:len() == 0 then
    save_vocabulary('source', data.dicts.src.words, opt.output_file .. '.src.dict')
  end

  if opt.targ_vocab_file:len() == 0 then
    save_vocabulary('target', data.dicts.targ.words, opt.output_file .. '.targ.dict')
  end

  if opt.features_vocabs_prefix:len() == 0 then
    save_features_vocabularies('source', data.dicts.src.features, opt.output_file)
    save_features_vocabularies('target', data.dicts.targ.features, opt.output_file)
  end

  print('Saving data to \'' .. opt.output_file .. '-train.t7\'...')
  torch.save(opt.output_file .. '-train.t7', data)

end

main()
