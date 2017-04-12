require('onmt.init')

local cmd = onmt.utils.ExtendedCmdLine.new('translate.lua')

local options = {
  {
    '-src', '',
    [[Source sequences to translate.]],
    {
      valid = onmt.utils.ExtendedCmdLine.nonEmpty
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
    '-idx_files', false,
    [[If set, source and target files are 'key value' with key match between source and target.]]
  }
}

cmd:setCmdLineOptions(options, 'Data')

onmt.translate.Translator.declareOpts(cmd)
onmt.utils.Cuda.declareOpts(cmd)
onmt.utils.Logger.declareOpts(cmd)

cmd:text('')
cmd:text('**Other options**')
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

  local goldReader
  local goldBatch

  local withGoldScore = opt.tgt:len() > 0

  if withGoldScore then
    goldReader = onmt.utils.FileReader.new(opt.tgt, opt.idx_files)
    goldBatch = {}
  end

  local outFile = io.open(opt.output, 'w')

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
    local srcSeq, srcSeqId = srcReader:next()

    local goldOutputSeq
    if withGoldScore then
      goldOutputSeq = goldReader:next()
    end

    if srcSeq ~= nil then
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
        if (srcBatch[b].words and #srcBatch[b].words == 0) then
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
              outFile:write(sentence .. '\n')
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
  outFile:close()
  _G.logger:shutDown()
end

main()
