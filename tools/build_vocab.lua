require('onmt.init')

local cmd = onmt.utils.ExtendedCmdLine.new('build_vocab.lua')

local options = {
  {
    '-data', '',
    [[Data file.]],
    {
      valid = onmt.utils.ExtendedCmdLine.fileExists
    }
  },
  {
    '-save_vocab', '',
    [[Vocabulary dictionary prefix.]],
    {
      valid = onmt.utils.ExtendedCmdLine.nonEmpty
    }
  },
  {
    '-vocab_size', { 50000 },
     [[List of source vocabularies size: `word[ feat1[ feat2[ ...] ] ]`.
      If = 0, vocabularies are not pruned.]]
  },
  {
    '-words_min_frequency', { 0 },
    [[List of source words min frequency: `word[ feat1[ feat2[ ...] ] ]`.
      If = 0, vocabularies are pruned by size.]]
  },
  {
    '-keep_frequency', false,
    [[Keep frequency of words in dictionary.]]
  },
  {
    '-idx_files', false,
    [[If set, each line of the data file starts with a first field which is the index of the sentence.]]
  }
}

cmd:setCmdLineOptions(options, 'Vocabulary')
onmt.utils.Logger.declareOpts(cmd)

local function isValid(sent)
  return #sent > 0
end

local function main()
  local opt = cmd:parse(arg)

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)

  local vocab = onmt.data.Vocabulary.init('source', opt.data, '', opt.vocab_size, opt.words_min_frequency, '', isValid, opt.keep_frequency, opt.idx_files)

  onmt.data.Vocabulary.save('source', vocab.words, opt.save_vocab .. '.dict')
  onmt.data.Vocabulary.saveFeatures('source', vocab.features, opt.save_vocab)

  _G.logger:shutDown()
end

main()
