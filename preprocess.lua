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
      enum = {'bitext', 'monotext', 'feattext'}
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

local function isValid(seq, maxSeqLength)
  if torch.isTensor(seq) then
    return seq:size(1) > 0 and seq:size(1) <= maxSeqLength
  end
  return #seq > 0 and #seq <= maxSeqLength
end

local function main()

  torch.manualSeed(opt.seed)

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)

  local Vocabulary = onmt.data.Vocabulary
  local Preprocessor = onmt.data.Preprocessor.new(opt, dataType)

  local data = { dataType=dataType }

  -- keep processing options in the structure for further traceability
  data.opt = opt

  data.dicts = {}

  _G.logger:info('Preparing vocabulary...')
  if dataType ~= 'feattext' then
    local src_file = opt.train_src
    if dataType == 'monotext' then
      src_file = opt.train
    end
    data.dicts.src = Vocabulary.init('source',
                                     src_file,
                                     opt.src_vocab or opt.vocab,
                                     opt.src_vocab_size or opt.vocab_size,
                                     opt.src_words_min_frequency or opt.words_min_frequency,
                                     opt.features_vocabs_prefix,
                                     function(s) return isValid(s, opt.src_seq_length or opt.seq_length) end,
                                     opt.idx_files,
                                     opt.keep_frequency)
  end
  if dataType ~= 'monotext' then
    local tgt_file = opt.train_tgt
    data.dicts.tgt = Vocabulary.init('target',
                                     tgt_file,
                                     opt.tgt_vocab,
                                     opt.tgt_vocab_size,
                                     opt.tgt_words_min_frequency,
                                     opt.features_vocabs_prefix,
                                     function(s) return isValid(s, opt.tgt_seq_length) end,
                                     opt.idx_files,
                                     opt.keep_frequency)
  end

  _G.logger:info('Preparing training data...')
  data.train = {}
  if dataType == 'monotext' then
    data.train.src = Preprocessor:makeMonolingualData(opt.train, data.dicts.src, isValid)
  elseif dataType == 'feattext' then
    data.train.src, data.train.tgt = Preprocessor:makeFeatTextData(opt.train_src, opt.train_tgt,
                                                                   data.dicts.tgt,
                                                                   isValid)
    -- record the size of the input layer
    data.dicts.srcInputSize = data.train.src.vectors[1]:size(2)
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
  elseif dataType == 'feattext' then
    data.valid.src, data.valid.tgt = Preprocessor:makeFeatTextData(opt.valid_src, opt.valid_tgt,
                                                                    data.dicts.tgt,
                                                                    isValid)
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
  elseif dataType == 'feattext' then
    if opt.tgt_vocab:len() == 0 then
      Vocabulary.save('target', data.dicts.tgt.words, opt.save_data .. '.tgt.dict')
    end
    if opt.features_vocabs_prefix:len() == 0 then
      Vocabulary.saveFeatures('target', data.dicts.tgt.features, opt.save_data)
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
