require('onmt.init')

local cmd = onmt.ExtendedCmdLine.new("preprocess.lua")

-- first argument define the dataType: BITEXT/MONO - default is BITEXT
local dataType = 'BITEXT'
for i=1,#arg do
  if arg[i]=='-data_type' and i<#arg then
    dataType = arg[i+1]
    break
  end
end

-------------- Options declaration
local preprocess_options = {
  {'-data_type',         'BITEXT',  [[Type of text to preprocess. Use 'MONO' for monolingual text.
                                    This option impacts all options choices.]],
                                    {enum={'BITEXT','MONO'}}},
  {'-save_data',               '',     [[Output file for the prepared data]]}
}

cmd:setCmdLineOptions(preprocess_options, "Preprocess")

onmt.data.Preprocessor.declareOpts(cmd, dataType)

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
  local Preprocessor = onmt.data.Preprocessor.new(opt, dataType)

  local data = { dataType=dataType }

  data.dicts = {}
  data.dicts.src = Vocabulary.init('train', opt.train, opt.vocab, opt.vocab_size,
                                   opt.features_vocabs_prefix, function(s) return isValid(s, opt.seq_length) end)
  if dataType ~= 'MONO' then
    data.dicts.tgt = Vocabulary.init('target', opt.train_tgt, opt.tgt_vocab, opt.tgt_vocab_size,
                                     opt.features_vocabs_prefix, function(s) return isValid(s, opt.src_seq_length) end)
  end

  _G.logger:info('Preparing training data...')
  data.train = {}
  if dataType == 'MONO' then
    data.train.src = Preprocessor:makeMonolingualData(opt.train, data.dicts.src, isValid)
  else
    data.train.src, data.train.tgt = Preprocessor:makeBilingualData(opt.train_src, opt.train_tgt,
                                                                    data.dicts.src, data.dicts.tgt,
                                                                    isValid)
  end

  _G.logger:info('')

  _G.logger:info('Preparing validation data...')
  data.valid = {}
  if dataType == 'MONO' then
    data.valid.src = Preprocessor:makeMonolingualData(opt.valid, data.dicts.src, isValid)
  else
    data.valid.src, data.valid.tgt = Preprocessor:makeBilingualData(opt.valid_src, opt.valid_tgt,
                                                                    data.dicts.src, data.dicts.tgt,
                                                                    isValid)
  end

  _G.logger:info('')

  if dataType == 'MONO' then
    if opt.vocab:len() == 0 then
      Vocabulary.save('source', data.dicts.src.words, opt.save_data .. '.dict')
    end
    if opt.features_vocabs_prefix:len() == 0 then
      Vocabulary.saveFeatures('source', data.dicts.src.features, opt.save_data)
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
