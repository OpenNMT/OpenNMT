local dict = require 's2sa.dict'

local file_reader = require 's2sa.file_reader'
local parallel_file_reader = require 's2sa.parallel_file_reader'

cmd = torch.CmdLine()

cmd:option('-train_src_file', '', [[Path to the training source data]])
cmd:option('-train_targ_file', '', [[Path to the training target data]])
cmd:option('-valid_src_file', '', [[Path to the validation source data]])
cmd:option('-valid_targ_file', '', [[Path to the validation target data]])

cmd:option('-output_file', '', [[Output file for the prepared data]])

cmd:option('-src_vocab_size', 50000, [[Size of the source vocabulary]])
cmd:option('-targ_vocab_size', 50000, [[Size of the target vocabulary]])

cmd:option('-seq_length', 50, [[Maximum sequence length]])
cmd:option('-shuffle', 1, [[Suffle data]])

cmd:option('-report_every', 100000, [[Report status every this many sentences]])

local opt = cmd:parse(arg)

local function reorder_table(tab, index)
  local new_tab = {}
  for i = 1, #tab do
    table.insert(new_tab, tab[index[i]])
  end
  return new_tab
end

local function make_vocabulary(filename, size)
  local vocab = dict.new({'<blank>', '<unk>', '<s>', '</s>'})
  local reader = file_reader.new(filename)

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
  vocab:prune(size)

  print('Created dictionary of size ' .. #vocab .. ' (pruned from ' .. original_size .. ')')

  return vocab
end

local function make_sentence(sent, dict)
  local vec = {}

  for i = 1, #sent do
    local idx = dict:lookup(sent[i])
    if idx == nil then
      idx = dict:lookup('<unk>')
    end
    table.insert(vec, idx)
  end

  return torch.IntTensor(vec)
end

local function make_data(src_file, targ_file, src_dict, targ_dict)
  local src = {}
  local targ = {}
  local sizes = {}

  local count = 0
  local ignored = 0

  local parallel_reader = parallel_file_reader.new(src_file, targ_file)

  while true do
    local src_tokens, targ_tokens = parallel_reader:next()
    if src_tokens == nil or targ_tokens == nil then
      break
    end
    if #src_tokens <= opt.seq_length and #targ_tokens <= opt.seq_length then
      table.insert(src, make_sentence(src_tokens, src_dict))
      table.insert(targ, make_sentence(targ_tokens, targ_dict))
      table.insert(sizes, #src_tokens)
    else
      ignored = ignored + 1
    end
    count = count + 1
    if count % opt.report_every == 0 then
      print('... ' .. count .. ' sentences prepared')
    end
  end

  if opt.shuffle == 1 then
    print('... shuffling sentences')
    local perm = torch.randperm(#src)
    src = reorder_table(src, perm)
    targ = reorder_table(targ, perm)
    sizes = reorder_table(sizes, perm)
  end

  print('... sorting sentences by size')
  local _, perm = torch.sort(torch.Tensor(sizes))
  src = reorder_table(src, perm)
  targ = reorder_table(targ, perm)

  print('Prepared ' .. #src .. ' sentences (' .. ignored .. ' ignored due to length > ' .. opt.seq_length .. ')')

  return {src, targ}
end

local function main()
  print('Building source vocabulary...')
  local src_dict = make_vocabulary(opt.train_src_file, opt.src_vocab_size)
  print('')

  print('Building target vocabulary...')
  local targ_dict = make_vocabulary(opt.train_targ_file, opt.targ_vocab_size)
  print('')

  print('Preparing training data...')
  local train_data = make_data(opt.train_src_file, opt.train_targ_file, src_dict, targ_dict)
  print('')

  print('Preparing validation data...')
  local valid_data = make_data(opt.valid_src_file, opt.valid_targ_file, src_dict, targ_dict)
  print('')

  local data = {train_data, valid_data}
  print('Saving data to ' .. opt.output_file .. '...')
  torch.save(opt.output_file, {data, src_dict, targ_dict})
end

main()
