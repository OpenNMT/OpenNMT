require('onmt.init')

local cmd = onmt.utils.ExtendedCmdLine.new('preprocess.lua')

-- First argument define the dataType: bitext/monotext - default is bitext.
local dataType = cmd.getArgument(arg, '-data_type') or 'bitext'

-- Options declaration
local options = {
  {
    '-data_type', 'bitext',
    [[Type of data to preprocess. Use 'monotext' for monolingual data.
      This option impacts all options choices.]],
    {
      enum = {'bitext', 'monotext'}
    }
  },
  {
    '-save_data', '',
    [[Output file for the prepared data.]],
    {
      valid = onmt.utils.ExtendedCmdLine.nonEmpty
    }
  }
}

cmd:setCmdLineOptions(options, 'Preprocess')

onmt.data.Preprocessor.declareOpts(cmd, dataType)
onmt.utils.Logger.declareOpts(cmd)

local otherOptions = {
  {
    '-seed', 3425,
    [[Random seed.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt()
    }
  }
}
cmd:setCmdLineOptions(otherOptions, 'Other')

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
  data.dicts.src = Vocabulary.init('source',
                                   opt.train_src or opt.train,
                                   opt.src_vocab or opt.vocab,
                                   opt.src_vocab_size or opt.vocab_size,
                                   opt.src_words_min_frequency or opt.words_min_frequency,
                                   opt.features_vocabs_prefix,
                                   function(s) return isValid(s, opt.src_seq_length or opt.seq_length) end,
                                   opt.keep_frequency)
  if dataType ~= 'monotext' then
    data.dicts.tgt = Vocabulary.init('target',
                                     opt.train_tgt,
                                     opt.tgt_vocab,
                                     opt.tgt_vocab_size,
                                     opt.tgt_words_min_frequency,
                                     opt.features_vocabs_prefix,
                                     function(s) return isValid(s, opt.tgt_seq_length) end,
                                     opt.keep_frequency)
  end

  _G.logger:info('Preparing training data...')
  data.train = {}
  if dataType == 'monotext' then
    data.train.src = Preprocessor:makeMonolingualData(opt.train, data.dicts.src, isValid)
  else
    data.train.src, data.train.tgt = Preprocessor:makeBilingualData(opt.train_src, opt.train_tgt,
                                                                    data.dicts.src, data.dicts.tgt,
                                                                    isValid)
  end

  _G.logger:info('')

  _G.logger:info('Preparing validation data...')
  data.valid = {}
  if dataType == 'monotext' then
    data.valid.src = Preprocessor:makeMonolingualData(opt.valid, data.dicts.src, isValid)
  else
    data.valid.src, data.valid.tgt = Preprocessor:makeBilingualData(opt.valid_src, opt.valid_tgt,
                                                                    data.dicts.src, data.dicts.tgt,
                                                                    isValid)
  end

  _G.logger:info('')

  if dataType == 'monotext' then
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
