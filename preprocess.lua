require('./lib/utils')

local constants = require('lib.constants')

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
cmd:option('-src_vocab_file', '', [[Pre-calculated source vocabulary]])
cmd:option('-targ_vocab_file', '', [[Pre-calculated target vocabulary]])

cmd:option('-seq_length', 50, [[Maximum sequence length]])
cmd:option('-shuffle', true, [[Suffle data]])
cmd:option('-seed', 3435, [[Random seed]])

cmd:option('-report_every', 100000, [[Report status every this many sentences]])

local opt = cmd:parse(arg)

local function make_vocabulary(filename, size)
  local vocab = utils.Dict.new({constants.PAD_WORD, constants.UNK_WORD,
                                constants.BOS_WORD, constants.EOS_WORD})
  local reader = utils.FileReader.new(filename)

  while true do
    local sent = reader:next()
    if sent == nil then
      break
    end
    for i = 1, #sent do
      vocab:add(sent[i])
    end
  end

  local original_size = #vocab

  reader:close()
  vocab = vocab:prune(size)

  print('Created dictionary of size ' .. #vocab .. ' (pruned from ' .. original_size .. ')')

  return vocab
end

local function init_vocabulary(name, data_file, vocab_file, vocab_size)
  local vocab

  if vocab_file:len() == 0 then
    print('Building ' .. name  .. ' vocabulary...')
    vocab = make_vocabulary(data_file, vocab_size)
  else
    print('Reading ' .. name .. ' vocabulary from \'' .. vocab_file .. '\'...')
    vocab = utils.Dict.new()
    vocab:load_file(vocab_file)
    print('Loaded ' .. #vocab .. ' ' .. name .. ' words')
  end

  print('')

  return vocab
end

local function save_vocabulary(name, vocab, file)
  print('Saving ' .. name .. ' vocabulary to \'' .. file .. '\'...')
  vocab:write_file(file)
end

local function make_data(src_file, targ_file, src_dict, targ_dict)
  local src = {}
  local targ = {}
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
      table.insert(src, src_dict:convert_to_idx(src_tokens, constants.UNK_WORD))
      table.insert(targ, targ_dict:convert_to_idx(targ_tokens, constants.UNK_WORD,
                                                  constants.BOS_WORD, constants.EOS_WORD))
      table.insert(sizes, #src_tokens)
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

  if opt.shuffle then
    print('... shuffling sentences')
    local perm = torch.randperm(#src)
    src = utils.Table.reorder(src, perm)
    targ = utils.Table.reorder(targ, perm)
    sizes = utils.Table.reorder(sizes, perm)
  end

  print('... sorting sentences by size')
  local _, perm = torch.sort(torch.Tensor(sizes))
  src = utils.Table.reorder(src, perm)
  targ = utils.Table.reorder(targ, perm)

  print('Prepared ' .. #src .. ' sentences (' .. ignored .. ' ignored due to length == 0 or > ' .. opt.seq_length .. ')')

  return src, targ
end

local function main()
  if opt.config:len() > 0 then
    opt = utils.Opt.load_config(opt.config, opt)
  end

  local required_options = {
    "train_src_file",
    "train_targ_file",
    "valid_src_file",
    "valid_targ_file",
    "output_file"
  }

  utils.Opt.require_options(opt, required_options)

  torch.manualSeed(opt.seed)

  local src_dict = init_vocabulary('source', opt.train_src_file, opt.src_vocab_file, opt.src_vocab_size)
  local targ_dict = init_vocabulary('target', opt.train_targ_file, opt.targ_vocab_file, opt.targ_vocab_size)

  print('Preparing training data...')
  local train_src, train_targ = make_data(opt.train_src_file, opt.train_targ_file, src_dict, targ_dict)
  print('')

  print('Preparing validation data...')
  local valid_src, valid_targ = make_data(opt.valid_src_file, opt.valid_targ_file, src_dict, targ_dict)
  print('')

  local data = {}
  data.train = {
    ["src"] = train_src,
    ["targ"] = train_targ
  }
  data.valid = {
    ["src"] = valid_src,
    ["targ"] = valid_targ
  }
  data.src_dict = src_dict
  data.targ_dict = targ_dict

  print('Saving data to \'' .. opt.output_file .. '-train.t7\'...')
  torch.save(opt.output_file .. '-train.t7', data)

  if opt.src_vocab_file:len() == 0 then
    save_vocabulary('source', src_dict, opt.output_file .. '.src.dict')
  end

  if opt.targ_vocab_file:len() == 0 then
    save_vocabulary('target', targ_dict, opt.output_file .. '.targ.dict')
  end

end

main()
