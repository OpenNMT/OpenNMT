require('onmt.init')

local cmd = onmt.utils.ExtendedCmdLine.new('translate.lua')

local options = {
  {'-src', '', [[Source sequence to decode (one line per sequence)]],
               {valid=onmt.utils.ExtendedCmdLine.nonEmpty}},
  {'-tgt', '', [[True target sequence (optional)]]},
  {'-output', 'pred.txt', [[Path to output the predictions (each line will be the decoded sequence)]]}
}

cmd:setCmdLineOptions(options, 'Data')

onmt.translate.Translator.declareOpts(cmd)

cmd:text('')
cmd:text('**Other options**')
cmd:text('')

cmd:option('-time', false, [[Measure batch translation time]])

onmt.utils.Cuda.declareOpts(cmd)
onmt.utils.Logger.declareOpts(cmd)

local function reportScore(name, scoreTotal, wordsTotal)
  _G.logger:info(name .. " AVG SCORE: %.2f, " .. name .. " PPL: %.2f",
                 scoreTotal / wordsTotal,
                 math.exp(-scoreTotal/wordsTotal))
end

local function main()
  local opt = cmd:parse(arg)

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)

  local srcReader = onmt.utils.FileReader.new(opt.src)
  local srcBatch = {}

  local goldReader
  local goldBatch

  local withGoldScore = opt.tgt:len() > 0

  if withGoldScore then
    goldReader = onmt.utils.FileReader.new(opt.tgt)
    goldBatch = {}
  end

  local translator = onmt.translate.Translator.new(opt)

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
    local srcSeq = srcReader:next()
    local goldOutputSeq
    if withGoldScore then
      goldOutputSeq = goldReader:next()
    end

    if srcSeq ~= nil then
      table.insert(srcBatch, translator:buildInput(srcSeq))

      if withGoldScore then
        table.insert(goldBatch, translator:buildInput(goldOutputSeq))
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
        if (#srcBatch[b].words == 0) then
          _G.logger:warning('Line ' .. sentId .. ' is empty.')
          outFile:write('\n')
        else
          _G.logger:info('SENT %d: %s', sentId, translator:buildOutput(srcBatch[b]))

          if withGoldScore then
            _G.logger:info('GOLD %d: %s', sentId, translator:buildOutput(goldBatch[b]), results[b].goldScore)
            _G.logger:info("GOLD SCORE: %.2f", results[b].goldScore)
            goldScoreTotal = goldScoreTotal + results[b].goldScore
            goldWordsTotal = goldWordsTotal + #goldBatch[b].words
          end

          for n = 1, #results[b].preds do
            local sentence = translator:buildOutput(results[b].preds[n])

            if n == 1 then
              outFile:write(sentence .. '\n')
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

        _G.logger:info('')
        sentId = sentId + 1
      end

      if srcSeq == nil then
        break
      end

      batchId = batchId + 1
      srcBatch = {}
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

  reportScore('PRED', predScoreTotal, predWordsTotal)

  if withGoldScore then
    reportScore('GOLD', goldScoreTotal, goldWordsTotal)
  end

  outFile:close()
  _G.logger:shutDown()
end

main()
