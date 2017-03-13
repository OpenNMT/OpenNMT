require('onmt.init')

local cmd = onmt.utils.ExtendedCmdLine.new('build_vocab.lua')

local options = {
  {'-data', '', 'Data file', {valid=onmt.utils.ExtendedCmdLine.fileExists}},
  {'-save_vocab', '', 'Vocabulary files name', {valid=onmt.utils.ExtendedCmdLine.nonEmpty}},
  {'-vocab_size', '50000', 'Comma-separated list of vocabularies size: word[,feat1,feat2,...]. If = 0, vocabularies are not pruned.',
   {valid=onmt.utils.ExtendedCmdLine.listUInt}},
  {'-words_min_frequency', '0',    [[Comma-separated list of words min frequency: word[,feat1,feat2,...]. If = 0, vocabularies are pruned by size.]],
   {valid=onmt.utils.ExtendedCmdLine.listUInt}}
}

cmd:setCmdLineOptions(options, 'Vocabulary creation')
onmt.utils.Logger.declareOpts(cmd)

local function isValid(sent)
  return #sent > 0
end

local function main()
  local opt = cmd:parse(arg)

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)

  local vocab = onmt.data.Vocabulary.init('source', opt.data, '', opt.vocab_size, opt.words_min_frequency, '', isValid)

  onmt.data.Vocabulary.save('source', vocab.words, opt.save_vocab .. '.dict')
  onmt.data.Vocabulary.saveFeatures('source', vocab.features, opt.save_vocab)

  _G.logger:shutDown()
end

main()
