require('onmt.init')

local cmd = onmt.utils.ExtendedCmdLine.new('tag.lua')

local options = {
  {
    '-src', '',
    [[Source sequences to tag.]],
    {
      valid = onmt.utils.ExtendedCmdLine.nonEmpty
    }
  },
  {
    '-output', 'pred.txt',
    [[Output file.]]
  }
}

cmd:setCmdLineOptions(options, 'Data')

onmt.tagger.Tagger.declareOpts(cmd)
onmt.utils.Cuda.declareOpts(cmd)
onmt.utils.Logger.declareOpts(cmd)

cmd:text('')
cmd:text('**Other options**')
cmd:text('')

cmd:option('-time', false, [[Measure average translation time.]])

local function main()
  local opt = cmd:parse(arg)

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)
  onmt.utils.Cuda.init(opt)

  local srcReader = onmt.utils.FileReader.new(opt.src)
  local srcBatch = {}

  local tagger = onmt.tagger.Tagger.new(opt)

  local outFile = io.open(opt.output, 'w')

  local sentId = 1
  local batchId = 1

  local timer
  if opt.time then
    timer = torch.Timer()
    timer:stop()
    timer:reset()
  end

  while true do
    local srcTokens = srcReader:next()

    if srcTokens ~= nil then
      table.insert(srcBatch, tagger:buildInput(srcTokens))
    elseif #srcBatch == 0 then
      break
    end

    if srcTokens == nil or #srcBatch == opt.batch_size then
      if opt.time then
        timer:resume()
      end

      local results = tagger:tag(srcBatch)

      if opt.time then
        timer:stop()
      end

      for b = 1, #results do
        if (#srcBatch[b].words == 0) then
          _G.logger:warning('Line ' .. sentId .. ' is empty.')
          outFile:write('\n')
        else
          _G.logger:info('SENT %d: %s', sentId, tagger:buildOutput(srcBatch[b]))

          local sentence = tagger:buildOutput(results[b])

          outFile:write(sentence .. '\n')

          _G.logger:info("PRED %d: %s", sentId, sentence)
        end

        _G.logger:info('')
        sentId = sentId + 1
      end

      if srcTokens == nil then
        break
      end

      batchId = batchId + 1
      srcBatch = {}
      collectgarbage()
    end
  end

  if opt.time then
    local time = timer:time()
    local sentenceCount = sentId-1
    _G.logger:info("Average sentence tagging time (in seconds):\n")
    _G.logger:info("avg real\t" .. time.real / sentenceCount .. "\n")
    _G.logger:info("avg user\t" .. time.user / sentenceCount .. "\n")
    _G.logger:info("avg sys\t" .. time.sys / sentenceCount .. "\n")
  end

  outFile:close()
  _G.logger:shutDown()
end

main()
