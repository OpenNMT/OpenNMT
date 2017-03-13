require('onmt.init')

local cmd = onmt.utils.ExtendedCmdLine.new('generate_metadata.lua')

local options = {
  {'-model', '', 'trained model file', {valid=onmt.utils.ExtendedCmdLine.nonEmpty}},
  {'-save_data', '', 'JSON metadata file', {valid=onmt.utils.ExtendedCmdLine.nonEmpty}},
  {'-name', '', 'name of the submitted system', {valid=onmt.utils.ExtendedCmdLine.nonEmpty}};
  {'-language_pair', '', 'language pair', {valid=onmt.utils.ExtendedCmdLine.nonEmpty}},
  {'-version', '', 'version of OpenNMT used for training'},
  {'-features', '', 'side features used'},
  {'-tokenization', '', 'side features used'},
  {'-encoder', '', 'encoder details'},
  {'-decoder', '', 'decoder details'},
  {'-oov', '', 'unknown replacement procedure'}
}

cmd:setCmdLineOptions(options, 'Model')

cmd:text('')
cmd:text('**Other options**')
cmd:text('')

onmt.utils.Logger.declareOpts(cmd)
onmt.utils.Cuda.declareOpts(cmd)

local function getLayerName(model, pattern)
  model:apply(function (m)
    if torch.typename(m):find(pattern) then
      return torch.typename(m)
    end
  end)
end

local function extractEpoch(modelFile)
  return modelFile:match('epoch%d+'):sub(6)
end

local function main()
  local opt = cmd:parse(arg)

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)

  onmt.utils.Cuda.init(opt)

  local checkpoint = torch.load(opt.model)

  local metadata = {}

  local function addField(name, description, defaultValue)
    metadata[name] = {}
    metadata[name].description = description
    metadata[name].value = defaultValue
  end

  addField('systemName', 'System name', opt.name)
  addField('constraint', 'Constrainted system', 'true')
  addField('framework', 'Framework', 'OpenNMT')
  addField('sourceLanguage', 'Source Language', opt.language_pair:sub(1, 2))
  addField('targetLanguage', 'Target Language', opt.language_pair:sub(3))
  addField('type', 'Type', 'NMT')
  addField('architecture', 'Global NN architecture', 'seq2seq-attn')
  addField('features', 'Use of side features', opt.features)
  addField('tokenization', 'Tokenization type', opt.tokenization)
  addField('vocabulary', 'Vocabulary size',
           checkpoint.dicts.src.words:size() .. '/' .. checkpoint.dicts.tgt.words:size())
  addField('layers', 'Number of layers', checkpoint.options.layers)
  addField('rnnType', 'RNN type', checkpoint.options.rnn_size .. ' ' .. (checkpoint.options.rnn_type or 'LSTM'))
  addField('dropout', 'Dropout', checkpoint.options.dropout)
  addField('embedding', 'Word Embedding', checkpoint.options.word_vec_size)
  addField('encoder', 'Encoder specific', opt.encoder)
  addField('decoder', 'Decoder specific', opt.decoder)
  addField('attention', 'Attention specific',
           getLayerName(checkpoint.models.decoder.modules[1], '.*[Aa]ttention.*'))
  addField('generator', 'Generator specific',
           getLayerName(checkpoint.models.decoder.modules[2], '.*[Ss]oftMax.*'))
  addField('oov', 'OOV Replacement', opt.oov)
  addField('optimization', 'Optimization', checkpoint.options.optim)
  addField('training', 'Training specific', extractEpoch(opt.model) .. ' epochs')

  local json = io.open(opt.save_data, 'w')
  local first = true

  json:write('{\n')
  for k, v in pairs(metadata) do
    if not first then
      json:write(',\n')
    else
      first = false
    end
    json:write('  "' .. k .. '": "' .. (v.value or '') .. '"')
  end
  json:write('\n}\n')

  json:close()
end

main()
