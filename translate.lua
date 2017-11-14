require('onmt.init')
local tokenizer = require 'tools.utils.tokenizer'
local normalizer = require('tools.utils.normalizer')
local BPE = require ('tools.utils.BPE')

local cmd = onmt.utils.ExtendedCmdLine.new('translate.lua')

local options = {
  {
    '-src', '',
    [[Source sequences to translate.]],
    {
      valid = onmt.utils.ExtendedCmdLine.fileExists
    }
  },
  {
    '-tgt', '',
    [[Optional true target sequences.]]
  },
  {
    '-output', 'pred.txt',
    [[Output file.]]
  },
  {
    '-save_attention', '',
    [[Optional attention output file.]]
  },
  {
    '-batch_size', 30,
    [[Batch size.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isInt(1)
    }
  },
  {
    '-idx_files', false,
    [[If set, source and target files are 'key value' with key match between source and target.]]
  }
}

cmd:setCmdLineOptions(options, 'Data')

onmt.translate.Translator.declareOpts(cmd)

  -- prepare tokenization option
local tok_options = {}
local topts = tokenizer.getOpts()
for _, v in ipairs(topts) do
  -- change mode option to include disabling mode (default)
  if v[1] == '-mode' then
    v = { '-mode', 'space',
        [[Define how aggressive should the tokenization be. `space` is space-tokenization.]],
          {
            enum = {'conservative', 'aggressive', 'space'}
          }
        }
  end
  local opt = onmt.utils.Table.deepCopy(v)
  opt[1] = '-tok_src_' .. v[1]:sub(2)
  table.insert(tok_options, opt)
  opt = onmt.utils.Table.deepCopy(v)
  opt[1] = '-tok_tgt_' .. v[1]:sub(2)
  table.insert(tok_options, opt)
end
cmd:setCmdLineOptions(tok_options, "Tokenizer")

onmt.utils.Cuda.declareOpts(cmd)
onmt.utils.Logger.declareOpts(cmd)

cmd:text('')
cmd:text('Other options')
cmd:text('')

cmd:option('-time', false, [[Measure average translation time.]])

local function reportScore(name, scoreTotal, wordsTotal)
  _G.logger:info(name .. " AVG SCORE: %.2f, " .. name .. " PPL: %.2f",
                 scoreTotal / wordsTotal,
                 math.exp(-scoreTotal/wordsTotal))
end

local function main()
  local opt = cmd:parse(arg)

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)
  _G.profiler = onmt.utils.Profiler.new()

  onmt.utils.Cuda.init(opt)

  local translator = onmt.translate.Translator.new(opt)

  local srcReader = onmt.utils.FileReader.new(opt.src, opt.idx_files, translator:srcFeat())
  local srcBatch = {}
  local srcIdBatch = {}

    -- tokenization options
  local tokenizers = { {}, {} }
  local normalizers = {}
  local bpes = {}
  for k, v in pairs(opt) do
    if k:sub(1,4) == 'tok_' then
      local idx = 1
      if k:sub(5, 8) == 'tgt_' then
        idx = 2
        k = k:sub(9)
      elseif k:sub(5,8) == 'src_' then
        k = k:sub(9)
      else
        k = k:sub(5)
      end
      tokenizers[idx][k] = v
    end
  end

  if opt.tok_src_bpe_model ~= '' then
     local myopt = {}
     myopt.bpe_model = opt.tok_src_bpe_model
     myopt.bpe_EOT_marker = opt.tok_src_bpe_EOT_marker
     myopt.bpe_BOT_marker = opt.tok_src_bpe_BOT_marker
     myopt.joiner_new = opt.tok_src_joiner_new
     myopt.joiner_annotate = opt.tok_src_joiner_annotate
     myopt.bpe_mode = opt.tok_src_bpe_mode
     myopt.bpe_case_insensitive = opt.tok_src_bpe_case_insensitive
     bpes[1] = BPE.new(myopt)
  end
  if opt.tok_tgt_bpe_model ~= '' then
     local myopt = {}
     myopt.bpe_model = opt.tok_tgt_bpe_model
     myopt.bpe_EOT_marker = opt.tok_tgt_bpe_EOT_marker
     myopt.bpe_BOT_marker = opt.tok_tgt_bpe_BOT_marker
     myopt.joiner_new = opt.tok_tgt_joiner_new
     myopt.joiner_annotate = opt.tok_sgt_joiner_annotate
     myopt.bpe_mode = opt.tok_tgt_bpe_mode
     myopt.bpe_case_insensitive = opt.tok_tgt_bpe_case_insensitive
     bpes[2] = BPE.new(myopt)
  end

  if opt.tok_src_normalize_cmd ~= '' then
    normalizers[1] = normalizer.new(opt.tok_src_normalize_cmd)
  end
  if opt.tok_tgt_normalize_cmd ~= '' then
    normalizers[2] = normalizer.new(opt.tok_tgt_normalize_cmd)
  end

  for i = 1, 2 do
    _G.logger:info("Using on-the-fly '%s' tokenization for input "..i, tokenizers[i]["mode"])
  end

  -- if source features - no tokenization
  if translator:srcFeat() then
    tokenizers[1] = nil
  end

  local goldReader
  local goldBatch

  local withGoldScore = opt.tgt:len() > 0
  local withAttention = opt.save_attention:len() > 0

  if withGoldScore then
    goldReader = onmt.utils.FileReader.new(opt.tgt, opt.idx_files)
    goldBatch = {}
  end

  local outFile = io.open(opt.output, 'w')
  local attFile
  if withAttention then
    attFile = io.open(opt.save_attention, 'w')
  end

  local sentId = 1
  local batchId = 1

  local predScoreTotal = 0
  local predWordsTotal = 0
  local goldScoreTotal = 0
  local goldWordsTotal = 0

  local timer
  if opt.time then
    timer = torch.Timer()
    timer:stop()
    timer:reset()
  end

  while true do
    local srcSeq, srcSeqId = srcReader:next(false)

    local goldOutputSeq
    if withGoldScore then
      goldOutputSeq = goldReader:next(false)
      if goldOutputSeq then
        if normalizers[2] then
          goldOutputSeq = normalizers[2]:normalize(goldOutputSeq)
        end
        goldOutputSeq = tokenizer.tokenize(tokenizers[2], goldOutputSeq, bpes[2])
      end
    end

    if srcSeq then
      if normalizers[1] then
        srcSeq = normalizers[1]:normalize(srcSeq)
      end
      if tokenizers[1] then
        srcSeq = tokenizer.tokenize(tokenizers[1], srcSeq, bpes[1])
      end
      table.insert(srcBatch, translator:buildInput(srcSeq))
      table.insert(srcIdBatch, srcSeqId)

      if withGoldScore then
        table.insert(goldBatch, translator:buildInputGold(goldOutputSeq))
      end
    elseif #srcBatch == 0 then
      break
    end

    if srcSeq == nil or #srcBatch == opt.batch_size then
      if opt.time then
        timer:resume()
      end

      local results = translator:translate(srcBatch, goldBatch)

      if opt.time then
        timer:stop()
      end

      for b = 1, #results do
        if (srcBatch[b].words and #srcBatch[b].words == 0
            or srcBatch[b].vectors and srcBatch[b].vectors:dim() == 0) then
          _G.logger:warning('Line ' .. sentId .. ' is empty.')
          outFile:write('\n')
        else
          if srcBatch[b].words then
            _G.logger:info('SENT %d: %s', sentId, translator:buildOutput(srcBatch[b]))
          else
            _G.logger:info('FEATS %d: IDX - %s - SIZE %d', sentId, srcIdBatch[b], srcBatch[b].vectors:size(1))
          end

          if withGoldScore then
            _G.logger:info('GOLD %d: %s', sentId, translator:buildOutput(goldBatch[b]), results[b].goldScore)
            _G.logger:info("GOLD SCORE: %.2f", results[b].goldScore)
            goldScoreTotal = goldScoreTotal + results[b].goldScore
            goldWordsTotal = goldWordsTotal + #goldBatch[b].words
          end

          if opt.dump_input_encoding then
            outFile:write(sentId, ' ', table.concat(torch.totable(results[b]), " "), '\n')
          else
            for n = 1, #results[b].preds do
              local sentence = translator:buildOutput(results[b].preds[n])
              sentence = tokenizer.detokenize(sentence, tokenizers[2])
              if normalizers[2] then
                sentence = normalizers[2]:normalize(sentence)
              end
              outFile:write(sentence .. '\n')

              if withAttention then
                local attentions = results[b].preds[n].attention
                local score = results[b].preds[n].score
                local targetLength = #attentions

                if translator:srcFeat() then
                  attFile:write(string.format('%d ||| %s ||| %f ||| %d\n',
                                              sentId, sentence, score, targetLength))
                else
                  local source = translator:buildOutput(srcBatch[b])
                  local sourceLength = #srcBatch[b].words
                  attFile:write(string.format('%d ||| %s ||| %f ||| %s ||| %d %d\n',
                                              sentId, sentence, score, source,
                                              sourceLength, targetLength))
                end

                for _, attention in ipairs(attentions) do
                  if attention ~= nil then
                    attFile:write(table.concat(torch.totable(attention), ' '))
                    attFile:write('\n')
                  end
                end

                attFile:write('\n')
              end

              if n == 1 then
                predScoreTotal = predScoreTotal + results[b].preds[n].score
                predWordsTotal = predWordsTotal + #results[b].preds[n].words

                if #results[b].preds > 1 then
                  _G.logger:info('')
                  _G.logger:info('BEST HYP:')
                end
              end

              if #results[b].preds > 1 then
                _G.logger:info("[%.2f] %s", results[b].preds[n].score, sentence)
              else
                _G.logger:info("PRED %d: %s", sentId, sentence)
                _G.logger:info("PRED SCORE: %.2f", results[b].preds[n].score)
              end
            end
          end
        end
        _G.logger:info('')
        sentId = sentId + 1
      end

      if srcSeq == nil then
        break
      end

      batchId = batchId + 1
      srcBatch = {}
      srcIdBatch = {}
      if withGoldScore then
        goldBatch = {}
      end
      collectgarbage()
    end
  end

  if opt.time then
    local time = timer:time()
    local sentenceCount = sentId-1
    _G.logger:info("Average sentence translation time (in seconds):\n")
    _G.logger:info("avg real\t" .. time.real / sentenceCount .. "\n")
    _G.logger:info("avg user\t" .. time.user / sentenceCount .. "\n")
    _G.logger:info("avg sys\t" .. time.sys / sentenceCount .. "\n")
  end

  if opt.dump_input_encoding == false then
    reportScore('PRED', predScoreTotal, predWordsTotal)

    if withGoldScore then
      reportScore('GOLD', goldScoreTotal, goldWordsTotal)
    end
  end

  if opt.save_beam_to:len() > 0 then
    translator:saveBeamHistories(opt.save_beam_to)
  end

  outFile:close()
  _G.logger:shutDown()
end

main()
