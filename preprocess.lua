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
      enum = {'bitext', 'monotext', 'feattext'},
      depends = function(opt) return opt.data_type ~= 'feattext' or opt.idx_files end
    }
  },
  {
    '-dry_run', false,
    [[If set, this will only prepare the preprocessor. Useful when using file sampling to
      test distribution rules.]]
  },
  {
    '-save_data', '',
    [[Output file for the prepared data.]],
    {
      depends = function(opt)
        return opt.dry_run or opt.save_data ~= '', "option `-save_data` is required"
      end
    }
  }
}

cmd:setCmdLineOptions(options, 'Preprocess')

onmt.data.Preprocessor.declareOpts(cmd, dataType)
onmt.utils.HookManager.declareOpts(cmd)
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

local function main()

  torch.manualSeed(opt.seed)

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)

  _G.hookManager = onmt.utils.HookManager.new(opt)

  local Preprocessor = onmt.data.Preprocessor.new(opt, dataType)

  if opt.dry_run then
    _G.logger:shutDown()
    return
  end

  local data = { dataType=dataType }

  -- keep processing options in the structure for further traceability
  data.opt = opt

  _G.logger:info('Preparing vocabulary...')
  data.dicts = Preprocessor:getVocabulary()

  _G.logger:info('Preparing training data...')
  data.train = Preprocessor:makeData('train', data.dicts)
  _G.logger:info('')

  _G.logger:info('Preparing validation data...')
  data.valid = Preprocessor:makeData('valid', data.dicts)
  _G.logger:info('')

  if dataType == 'monotext' then
    if opt.vocab:len() == 0 then
      onmt.data.Vocabulary.save('source', data.dicts.src.words, opt.save_data .. '.dict')
    end
    if opt.features_vocabs_prefix:len() == 0 then
      onmt.data.Vocabulary.saveFeatures('source', data.dicts.src.features, opt.save_data)
    end
  elseif dataType == 'feattext' then
    if opt.tgt_vocab:len() == 0 then
      onmt.data.Vocabulary.save('target', data.dicts.tgt.words, opt.save_data .. '.tgt.dict')
    end
    if opt.features_vocabs_prefix:len() == 0 then
      onmt.data.Vocabulary.saveFeatures('target', data.dicts.tgt.features, opt.save_data)
    end
  else
    if opt.src_vocab:len() == 0 then
      onmt.data.Vocabulary.save('source', data.dicts.src.words, opt.save_data .. '.src.dict')
    end

    if opt.tgt_vocab:len() == 0 then
      onmt.data.Vocabulary.save('target', data.dicts.tgt.words, opt.save_data .. '.tgt.dict')
    end
    if opt.features_vocabs_prefix:len() == 0 then
      onmt.data.Vocabulary.saveFeatures('source', data.dicts.src.features, opt.save_data..'.source')
      onmt.data.Vocabulary.saveFeatures('target', data.dicts.tgt.features, opt.save_data..'.target')
    end
  end

  _G.logger:info('Saving data to \'' .. opt.save_data .. '-train.t7\'...')
  torch.save(opt.save_data .. '-train.t7', data, 'binary', false)
  _G.logger:shutDown()
end

main()
