require('onmt.init')

local cmd = onmt.utils.ExtendedCmdLine.new('build_dict.lua')

local options = {
  {'-data', '', 'Data file', {valid=onmt.utils.ExtendedCmdLine.fileExists}},
  {'-save_dict', '', 'Dictionary file', {valid=onmt.utils.ExtendedCmdLine.nonEmpty}},
  {'-vocab_size', '50000', 'Comma-separated list of target vocabularies size: word[,feat1,feat2,...].',
   {valid=onmt.utils.ExtendedCmdLine.listUInt}}
}

cmd:setCmdLineOptions(options, 'Dictionary creation')
onmt.utils.Logger.declareOpts(cmd)

local function isValid(sent)
  return #sent > 0
end

local function main()
  local opt = cmd:parse(arg)

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)

  local vocab = onmt.data.Vocabulary.init('source', opt.data, '', opt.vocab_size, '', isValid)

  onmt.data.Vocabulary.save('source', vocab.words, opt.save_dict .. '.dict')
  onmt.data.Vocabulary.saveFeatures('source', vocab.features, opt.save_dict)

  _G.logger:shutDown()
end

main()
