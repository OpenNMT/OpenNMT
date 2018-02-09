require('torch')
require('onmt.init')

local tokenizer = require('tools.utils.tokenizer')
local unicode = require('tools.utils.unicode')
local separators = require('tools.utils.separators')
local HookManager = require('onmt.utils.HookManager')
local tds = require('tds')

local cmd = onmt.utils.ExtendedCmdLine.new('learn_bpe.lua')

local options = {
  {
    '-size', '30000',
    [[The number of merge operations to learn.]]
  },
  {
    '-bpe_mode', 'suffix',
    [[Define the BPE mode.
      `prefix`: append `<w>` to the begining of each word to learn prefix-oriented pair statistics;
      `suffix`: append `</w>` to the end of each word to learn suffix-oriented pair statistics,
       as in the original Python script;
      `both`: `suffix` and `prefix`;
      `none`: no `suffix` nor `prefix`.]],
    {
      enum = {'suffix', 'prefix', 'both', 'none'}
    }
  },
  {
    '-bpe_EOT_marker', separators.EOT,
    [[Marker used to mark the End of Token while applying BPE in mode 'prefix' or 'both'.]]
  },
  {
    '-bpe_BOT_marker', separators.BOT,
    [[Marker used to mark the Beginning of Token while applying BPE in mode 'suffix' or 'both'.]]
  },
  {
    '-save_bpe', '',
    [[Path to save the output model.]],
    {
      valid = onmt.utils.ExtendedCmdLine.nonEmpty
    }
  }
}

cmd:setCmdLineOptions(options, 'BPE')

-- prepare tokenization option
options = {}
local topts = tokenizer.getOpts()
for _, v in ipairs(topts) do
  if v[1]:sub(1,4)  ~= '-bpe' then
    -- change mode option to include disabling mode (default)
    if v[1] == '-mode' then
      v[2] = 'space'
    end
    local opttmp = {table.unpack(v)}
    opttmp[1] = '-tok_' .. v[1]:sub(2)
    table.insert(options, {table.unpack(opttmp)})
  end
end

cmd:setCmdLineOptions(options, "Tokenizer")

-- insert on the fly the option depending if there is a hook selected
onmt.utils.HookManager.updateOpt(arg, cmd)
onmt.utils.HookManager.declareOpts(cmd)

onmt.utils.Logger.declareOpts(cmd)

local opt = cmd:parse(arg)

local function string2word(s)
  local t = {}
  if opt.bpe_mode == 'prefix' or opt.bpe_mode == 'both' then table.insert(t, opt.bpe_BOT_marker) end
  if s:sub(1, separators.ph_marker_open:len()) == separators.ph_marker_open then
    table.insert(t, s)
  else
    for _, c in unicode.utf8_iter(s) do
      table.insert(t, c)
    end
  end
  if opt.bpe_mode == 'suffix' or opt.bpe_mode == 'both' then table.insert(t, opt.bpe_EOT_marker) end
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

local function updatedict(d, key1, key2, value)
  if d[key1] == nil then
    d[key1] = tds.Hash({ [key2] = value})
  else
    if d[key1][key2] == nil then
      d[key1][key2] = 1
    else
      d[key1][key2] = d[key1][key2] + value
    end
  end
end

local function get_vocabulary()
  local vocab = tds.Hash()
  local l = io.read()

  -- tokenization options
  local tokopts = {}
  for k, v in pairs(opt) do
    if k:sub(1,4) == 'tok_' then
      k = k:sub(5)
      tokopts[k] = v
    end
  end
  _G.logger:info("Using on-the-fly '%s' tokenization for input", tokopts["mode"])

  local segmentor = function (line) return tokenizer.tokenize(tokopts, line, nil) end

  _G.logger:info('Building vocabulary from STDIN')
  local count = 1
  while not(l == nil) do
    local toks = segmentor(l)
    local words = onmt.utils.Features.extract(toks)
    for i = 1, #words do
      local word = words[i]
      vocab[word] = (vocab[word] or 0) + 1
    end
    l = io.read()
    count = count + 1
    if count % 100000 == 0 then _G.logger:info('... ' .. count .. ' sentences processed') end
  end
  local vocabreal = tds.Hash()
  for k, v in pairs(vocab) do
    vocabreal[string2word(k)] = v
  end
  return vocabreal
end

local function get_pair_statistics(vocab)
  local stats = tds.Hash()
  local indices = tds.Hash()
  for idx, word_freq in ipairs(vocab) do
    local word = word_freq[1]
    local freq = word_freq[2]
    local chars = string.split (word, " ")
    local prev_char = chars[1]
    for i=2, #chars do
      local bigram = prev_char .. " " .. chars[i]
      stats[bigram] = ( stats[bigram] or 0 ) + freq
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
      vocab[idx] = tds.Vec({new_word, freq})
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
  indices[pair] = tds.Hash()
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
          stats[prev] = ( stats[prev] or 0 ) - freq
          updatedict(indices, prev, idx, -1)
        end
        if i <= #old_word-2 then
          if old_word[i+2] ~= first or i >= #old_word-3 or old_word[i+3] ~= second then
            local nex = old_word[i+1] .. " " .. old_word[i+2]
            stats[nex] = ( stats[nex] or 0 ) - freq
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
        stats[prev] = ( stats[prev] or 0 ) + freq
        updatedict(indices, prev, idx, 1)
      end
      if i <= #new_word-1 and new_word[i+1] ~= new_pair then
        local nex = new_word[i] .. " " .. new_word[i+1]
        stats[nex] = ( stats[nex] or 0 ) + freq
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

local function prune_stats(stats, big_stats, threshold)
  for item, freq in pairs(stats) do
    if freq < threshold then
      stats[item] = nil
      if freq < 0 then
        big_stats[item] = ( big_stats[item] or 0 ) + freq
      else
        big_stats[item] = freq
      end
    end
  end
end

local function clone (t) -- shallow-copy a tds hash
    local target = tds.Hash()
    for k, v in pairs(t) do target[k] = v end
    return target
end

local function main()

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level, opt.log_tag)
  _G.hookManager = HookManager.new(opt)

  local vocab = get_vocabulary()
  local sorted_vocab = tds.Vec()
  for k, v in pairs(vocab) do sorted_vocab[#sorted_vocab+1] = tds.Vec({k, v}) end
  sorted_vocab:sort(function(a, b) return a[2] > b[2] end)

  _G.logger:info('Getting pair statistics from vocabulary')
  local stats, indices = get_pair_statistics (sorted_vocab)

  local big_stats = clone(stats)
  local threshold = stats[maxKey(stats)] / 10

  local bpe_options = {'v3'}
  if opt.bpe_mode == 'prefix' or opt.bpe_mode == 'both' then table.insert(bpe_options, "true") else table.insert(bpe_options, "false") end
  if opt.bpe_mode == 'suffix' or opt.bpe_mode == 'both' then table.insert(bpe_options, "true") else table.insert(bpe_options, "false") end
  if opt.tok_case_feature then table.insert(bpe_options, "true") else table.insert(bpe_options, "false") end
  table.insert(bpe_options, opt.bpe_BOT_marker)
  table.insert(bpe_options, opt.bpe_EOT_marker)

  _G.logger:info('Generating merge operations to output')

  local f = assert(io.open(opt.save_bpe, 'w'))
  f:write(table.concat(bpe_options, ";") .. "\n")

  for i = 1, opt.size do
    local most_frequent
    if stats ~= nil then
      most_frequent = maxKey(stats)
    end

    -- we probably missed the best pair because of pruning; go back to full statistics

    if stats == nil or stats[most_frequent] < threshold then
      prune_stats(stats, big_stats, threshold)
      stats = clone(big_stats)
      most_frequent = maxKey(stats)

      -- threshold is inspired by Zipfian assumption, but should only affect speed
      threshold = stats[most_frequent] * i/(i+10000.0)
      prune_stats(stats, big_stats, threshold)
    end

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
