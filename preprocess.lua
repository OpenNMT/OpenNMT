require('onmt.init')

local cmd = onmt.ExtendedCmdLine.new("preprocess.lua")

local mode = 'BITEXT'
if #arg>0 and arg[1]=='MONO' then
  mode = 'MONO'
end

-------------- Options declaration
local preprocess_options = {
  {'-save_data',               '',     [[Output file for the prepared data]]}
}

cmd:setCmdLineOptions(preprocess_options, "Preprocess")

onmt.data.Preprocessor.declareOpts(cmd, mode)

local misc_options = {
  {'-seed',                   3425,    [[Random seed]],
                                   {valid=onmt.ExtendedCmdLine.isUInt()}},
  {'-report_every',           100000,  [[Report status every this many sentences]],
                                   {valid=onmt.ExtendedCmdLine.isUInt()}}
}
cmd:setCmdLineOptions(misc_options, "Other")
onmt.utils.Logger.declareOpts(cmd)

local opt = cmd:parse(arg)

local function isValid(sent, maxSeqLength)
  return #sent > 0 and #sent <= maxSeqLength
end

local function main()

  torch.manualSeed(opt.seed)

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)

  local Vocabulary = onmt.data.Vocabulary
  local Preprocessor = onmt.data.Preprocessor.new(opt, mode)

  local data = { mode=mode }

  if mode == 'MONO' then
    data.dicts = Vocabulary.init('train', opt.train, opt.vocab, opt.vocab_size,
                               opt.features_vocabs_prefix, function(s) return isValid(s, opt.seq_length) end)
  else
    data.dicts.src = Vocabulary.init('source', opt.train_src, opt.src_vocab, opt.src_vocab_size,
                                     opt.features_vocabs_prefix, function(s) return isValid(s, opt.src_seq_length) end)
    data.dicts.tgt = Vocabulary.init('target', opt.train_tgt, opt.tgt_vocab, opt.tgt_vocab_size,
                                     opt.features_vocabs_prefix, function(s) return isValid(s, opt.src_seq_length) end)
  end

  _G.logger:info('Preparing training data...')
  if mode == 'MONO' then
    data.train = Preprocessor:makeMonolingualData(opt.train, data.dicts, isValid)
  else
    data.train = {}
    data.train.src, data.train.tgt = Preprocessor:makeBilingualData(opt.train_src, opt.train_tgt,
                                                                    data.dicts.src, data.dicts.tgt,
                                                                    isValid)
  end

  _G.logger:info('')

  _G.logger:info('Preparing validation data...')
  if mode == 'MONO' then
      data.valid = Preprocessor:makeMonolingualData(opt.valid, data.dicts, isValid)
  else
    data.valid = {}
    data.valid.src, data.valid.tgt = Preprocessor:makeBilingualData(opt.valid_src, opt.valid_tgt,
                                                                    data.dicts.src, data.dicts.tgt,
                                                                    isValid)
  end

  _G.logger:info('')

  if mode == 'MONO' then
    if opt.vocab:len() == 0 then
      Vocabulary.save('train', data.dicts.words, opt.save_data .. '.dict')
    end
    if opt.features_vocabs_prefix:len() == 0 then
      Vocabulary.saveFeatures('train', data.dicts.features, opt.save_data)
    end
  else
    if opt.src_vocab:len() == 0 then
      Vocabulary.save('source', data.dicts.src.words, opt.save_data .. '.src.dict')
    end

    if opt.tgt_vocab:len() == 0 then
      Vocabulary.save('target', data.dicts.tgt.words, opt.save_data .. '.tgt.dict')
    end
    if opt.features_vocabs_prefix:len() == 0 then
      Vocabulary.saveFeatures('source', data.dicts.src.features, opt.save_data)
      Vocabulary.saveFeatures('target', data.dicts.tgt.features, opt.save_data)
    end
  end

  _G.logger:info('Saving data to \'' .. opt.save_data .. '-train.t7\'...')
  torch.save(opt.save_data .. '-train.t7', data, 'binary', false)
  _G.logger:shutDown()
end

main()
