require('torch')

local Logger = require('onmt.utils.Logger')
local unicode = require 'tools.utils.unicode'
local tokenizer = require('tools.utils.tokenizer')
local case = require ('tools.utils.case')
local separators = require('tools.utils.separators')

local cmd = torch.CmdLine()

cmd:text("")
cmd:text("**learn_bpe.lua**")
cmd:text("")

cmd:option('-size', '30000', [[The number of merge operations to learn]])
cmd:option('-t', false, [[tokenize the input with tokenizer, the same options as tokenize.lua, but only '-mode' is taken into account for BPE training]])
cmd:option('-mode', 'conservative', [[Define how aggressive should the tokenization be - 'aggressive' only keeps sequences of letters/numbers, 'conservative' allows mix of alphanumeric as in: '2,000', 'E65', 'soft-landing']])
cmd:option('-lc', false, [[lowercase the output from the tokenizer before BPE learning]])
cmd:option('-bpe_mode', 'suffix', [[Define the mode for bpe - 'prefix': Append '﹤' to the begining of each word to learn prefix-oriented pair statistics;
'suffix': Append '﹥' to the end of each word to learn suffix-oriented pair statistics, as in the original python script;
'both': suffix and prefix; 'none': no suffix nor prefix]])
cmd:option('-save_bpe', '', [[Path to save the output model]])

--Logger.declareOpts(cmd)

local opt = cmd:parse(arg)

local function string2word(s)
  local t = {}
  if opt.bpe_mode == 'prefix' or opt.bpe_mode == 'both' then table.insert(t, separators.BOT) end
  for _, c in unicode.utf8_iter(s) do
    if c == separators.BOT then c = separators.BOT_substitute end
    if c == separators.EOT then c = separators.EOT_substitute end
    table.insert(t, c)
  end
  if opt.bpe_mode == 'suffix' or opt.bpe_mode == 'both' then table.insert(t, separators.EOT) end
  return table.concat(t, " ")
end

local function replace(word, bigram)
  local new_word = {}
  local merge = false
  for i = 1, #word do
    if (merge) then
      if word[i] == bigram[2] then
        table.insert(new_word, bigram[1] .. bigram[2])
        merge = false
      elseif word[i] == bigram[1] then
        table.insert(new_word, bigram[1])
      else
        table.insert(new_word, bigram[1])
        table.insert(new_word, word[i])
        merge = false
      end
    else
      if bigram[1] == word[i] then
        merge = true
      else
        table.insert(new_word, word[i])
      end
    end
  end
  if merge then table.insert(new_word, word[#word]) end
  return table.concat(new_word, " ")
end

local function defaultdict(dvalue)
  local tbl = {}
  local mtbl = {}
  mtbl.__index = function(t, key)
    local val = rawget(t, key)
    return val or dvalue
  end
  setmetatable(tbl, mtbl)
  return tbl
end

local function updatedict(d, key1, key2, value)
  if d[key1] == nil then
    d[key1] = { [key2] = value}
  else
    if d[key1][key2] == nil then
      d[key1][key2] = 1
    else
      d[key1][key2] = d[key1][key2] + value
    end
  end
end

local function get_vocabulary()
  local vocab = defaultdict(0)
  local l = io.read()

  local segmentor = function (a) return string.split(a, " ") end
  if opt.t then
    segmentor = function (a) return tokenizer.tokenize(opt, a) end
    if opt.lc then
      segmentor = function (a) return case.lowerCase(tokenizer.tokenize(opt, a)) end
    end
  elseif opt.lc or opt.mode == 'aggressive' then
    _G.logger:warning('The tokenization options -lc or -mode are disabled, add -t to cmd to enable these options')
  end
  _G.logger:info('Building vocabulary from STDIN')
  local count = 1
  while not(l == nil) do
    local toks = segmentor(l)
    for i = 1, #toks do
      local word = toks[i]
      vocab[word] = vocab[word] + 1
    end
    l=io.read()
    count = count + 1
    if count % 100000 == 0 then _G.logger:info('... ' .. count .. ' sentences processed') end
  end
  local vocabreal = {}
  for k, v in pairs(vocab) do
    vocabreal[string2word(k)] = v
  end
  return vocabreal
end

local function get_pair_statistics(vocab)
  local stats = defaultdict(0)
  local indices = {}
  for idx, word_freq in ipairs(vocab) do
    local word = word_freq[1]
    local freq = word_freq[2]
    local chars = string.split (word, " ")
    local prev_char = chars[1]
    for i=2, #chars do
      local bigram = prev_char .. " " .. chars[i]
      stats[bigram] = stats[bigram] + freq
      updatedict(indices, bigram, idx, 1)
      prev_char = chars[i]
    end
  end
  return stats, indices
end

local function replace_pair(pair, vocab, indices)
  local changed = {}
  local bigram = string.split (pair, " ")

  for idx, ifreq in pairs (indices[pair]) do
    if not (ifreq < 1) then
      local word_freq = vocab[idx]
      local word = word_freq[1]
      local freq = word_freq[2]
      local new_word = replace(string.split(word, ' '), bigram)
      vocab[idx] = {new_word, freq}
      table.insert(changed, {idx, new_word, word, freq})
    end
  end
  return changed
end

local function find_index (word, pattern, i)
  for k,v in pairs(word) do
    if k >= i and v == pattern then return k end
  end
end

local function update_pair_statistics(pair, changed, stats, indices)
  stats[pair] = 0
  indices[pair] = {}
  local bigram = string.split (pair, " ")
  local first = bigram[1]
  local second = bigram[2]
  local new_pair = first .. second
  for ii = 1, #changed do
    local c = changed[ii]
    local idx = c [1]
    local new_word = string.split(c[2], " ")
    local old_word = string.split(c[3], " ")
    local freq = c[4]

    local i = 1

    while true do
      i = find_index(old_word, first, i)
      if i == nil then break end
      if i < #old_word and old_word[i+1] == second then
        if i > 1 then
          local prev = old_word[i-1] .. " " .. old_word[i]
          stats[prev] = stats[prev] - freq
          updatedict(indices, prev, idx, -1)
        end
        if i <= #old_word-2 then
          if old_word[i+2] ~= first or i >= #old_word-3 or old_word[i+3] ~= second then
            local nex = old_word[i+1] .. " " .. old_word[i+2]
            stats[nex] = stats[nex] - freq
            updatedict(indices, nex, idx, -1)
          end
        end
        i = i + 2
      else
        i = i + 1
      end
    end
    i = 0
    while true do
      i = find_index(new_word, new_pair, i)
      if i == nil then break end
      if i > 1 then
        local prev = new_word[i-1] .. " " .. new_word[i]
        stats[prev] = stats[prev] + freq
        updatedict(indices, prev, idx, 1)
      end
      if i <= #new_word-1 and new_word[i+1] ~= new_pair then
        local nex = new_word[i] .. " " .. new_word[i+1]
        stats[nex] = stats[nex] + freq
        updatedict(indices, nex, idx, 1)
      end
      i = i + 1
    end
  end
end

local function maxKey(map)
  local max_value = 0
  local max_key = {}
  for k, v in pairs(map) do
    if v > max_value then
      max_value = v
      max_key = {k}
    elseif v == max_value then
      table.insert(max_key, k)
    end
  end
  table.sort(max_key)
  return max_key[1]
end

local function main()

  _G.logger = Logger.new(opt.log_file, opt.disable_logs, opt.log_level)

  local bpe_options = {}
  if opt.bpe_mode == 'prefix' or opt.bpe_mode == 'both' then table.insert(bpe_options, "true") else table.insert(bpe_options, "false") end
  if opt.bpe_mode == 'suffix' or opt.bpe_mode == 'both' then table.insert(bpe_options, "true") else table.insert(bpe_options, "false") end
  if opt.lc and opt.t then table.insert(bpe_options, "true") else table.insert(bpe_options, "false") end
  table.insert(bpe_options, opt.mode)

  local vocab = get_vocabulary()
  local sorted_vocab = {}
  for k, v in pairs(vocab) do sorted_vocab[#sorted_vocab+1] = {k, v} end
  table.sort(sorted_vocab, function(a, b) return a[2] > b[2] end)

  _G.logger:info('Geting pair statistics from vocabulary')
  local stats, indices = get_pair_statistics (sorted_vocab)

  _G.logger:info('Generating merge operations to output')

  local f = assert(io.open(opt.save_bpe, 'w'))
  f:write(table.concat(bpe_options, ";") .. "\n")

  for i = 1, opt.size do
    local most_frequent = maxKey(stats)
    if stats[most_frequent] < 2 then
      io.stderr:write("No pair has frequency > 1. Stopping\n")
      break
    end
    local changed = replace_pair(most_frequent, sorted_vocab, indices)
    update_pair_statistics(most_frequent, changed, stats, indices)
    stats[most_frequent] = 0
    f:write(most_frequent .. "\n")
    if i % 1000 == 0 then _G.logger:info('... ' .. i .. ' merge operations generated') end
  end
  f:close()
end

main()
