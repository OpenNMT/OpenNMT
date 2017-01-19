require('onmt.init')

local cmd = torch.CmdLine()

cmd:text("")
cmd:text("**onmt.translate.lua**")
cmd:text("")


cmd:option('-config', '', [[Read options from this file]])

cmd:text("")
cmd:text("**Data options**")
cmd:text("")

cmd:option('-src', '', [[Source sequence to decode (one line per sequence)]])
cmd:option('-tgt', '', [[True target sequence (optional)]])
cmd:option('-output', 'pred.txt', [[Path to output the predictions (each line will be the decoded sequence]])

onmt.translate.Translator.declareOpts(cmd)

cmd:text("")
cmd:text("**Other options**")
cmd:text("")
cmd:option('-time', false, [[Measure batch translation time]])

onmt.utils.Cuda.declareOpts(cmd)
onmt.utils.Logger.declareOpts(cmd)

local function reportScore(name, scoreTotal, wordsTotal)
  _G.logger:info(name .. " AVG SCORE: %.4f, " .. name .. " PPL: %.4f",
                 scoreTotal / wordsTotal,
                 math.exp(-scoreTotal/wordsTotal))
end

local function main()
  local opt = cmd:parse(arg)

  local requiredOptions = {
    "model",
    "src"
  }

  onmt.utils.Opt.init(opt, requiredOptions)

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)

  local srcReader = onmt.utils.FileReader.new(opt.src)
  local srcBatch = {}
  local srcWordsBatch = {}
  local srcFeaturesBatch = {}

  local tgtReader
  local tgtBatch
  local tgtWordsBatch
  local tgtFeaturesBatch

  local withGoldScore = opt.tgt:len() > 0

  if withGoldScore then
    tgtReader = onmt.utils.FileReader.new(opt.tgt)
    tgtBatch = {}
    tgtWordsBatch = {}
    tgtFeaturesBatch = {}
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
    local srcTokens = srcReader:next()
    local tgtTokens
    if withGoldScore then
      tgtTokens = tgtReader:next()
    end

    if srcTokens ~= nil then
      local srcWords, srcFeats = onmt.utils.Features.extract(srcTokens)
      table.insert(srcBatch, srcTokens)
      table.insert(srcWordsBatch, srcWords)
      if #srcFeats > 0 then
        table.insert(srcFeaturesBatch, srcFeats)
      end

      if withGoldScore then
        local tgtWords, tgtFeats = onmt.utils.Features.extract(tgtTokens)
        table.insert(tgtBatch, tgtTokens)
        table.insert(tgtWordsBatch, tgtWords)
        if #tgtFeats > 0 then
          table.insert(tgtFeaturesBatch, tgtFeats)
        end
      end
    elseif #srcBatch == 0 then
      break
    end

    if srcTokens == nil or #srcBatch == opt.batch_size then
      if opt.time then
        timer:resume()
      end

      local predBatch, info = translator:translate(srcWordsBatch, srcFeaturesBatch,
                                                   tgtWordsBatch, tgtFeaturesBatch)

      if opt.time then
        timer:stop()
      end

      for b = 1, #predBatch do
        local srcSent = table.concat(srcBatch[b], " ")
        local predSent = table.concat(predBatch[b], " ")

        outFile:write(predSent .. '\n')

        if (#srcBatch[b] == 0) then
          _G.logger:warning('SENT ' .. sentId .. ' is empty.')
        else
          _G.logger:info('SENT ' .. sentId .. ': ' .. srcSent)
          _G.logger:info('PRED ' .. sentId .. ': ' .. predSent)
          _G.logger:info("PRED SCORE: %.4f", info[b].score)

          predScoreTotal = predScoreTotal + info[b].score
          predWordsTotal = predWordsTotal + #predBatch[b]

          if withGoldScore then
            local tgtSent = table.concat(tgtBatch[b], " ")

            _G.logger:info('GOLD ' .. sentId .. ': ' .. tgtSent)
            _G.logger:info("GOLD SCORE: %.4f", info[b].goldScore)

            goldScoreTotal = goldScoreTotal + info[b].goldScore
            goldWordsTotal = goldWordsTotal + #tgtBatch[b]
          end

          if opt.n_best > 1 then
            _G.logger:info('\nBEST HYP:')
            for n = 1, #info[b].nBest do
              local nBest = table.concat(info[b].nBest[n].tokens, " ")
              _G.logger:info("[%.4f] %s", info[b].nBest[n].score, nBest)
            end
          end
        end

        _G.logger:info('')
        sentId = sentId + 1
      end

      if srcTokens == nil then
        break
      end

      batchId = batchId + 1
      srcBatch = {}
      srcWordsBatch = {}
      srcFeaturesBatch = {}
      if withGoldScore then
        tgtBatch = {}
        tgtWordsBatch = {}
        tgtFeaturesBatch = {}
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
