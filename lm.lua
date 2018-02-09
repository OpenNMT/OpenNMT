require('onmt.init')

local cmd = onmt.utils.ExtendedCmdLine.new('lm.lua')

local options = {
  {
    'mode', 'string',
    [['score' apply lm to input text, 'sample' samples output based on input text.]],
    {
      enum = { 'score', 'sample' }
    }
  },
  {
    '-src', '',
    [[Source sequences to sample/score.]],
    {
      valid = onmt.utils.ExtendedCmdLine.nonEmpty
    }
  },
  {
    '-output', 'output.txt',
    [[Output file depend on `<mode>`.]]
  }
}

cmd:setCmdLineOptions(options, 'Data')

onmt.lm.LM.declareOpts(cmd)
onmt.utils.Cuda.declareOpts(cmd)
onmt.utils.Logger.declareOpts(cmd)

cmd:text('')
cmd:text('Other options')
cmd:text('')

cmd:option('-time', false, [[Measure average translation time.]])

local function main()
  local opt = cmd:parse(arg)

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level, opt.log_tag)
  onmt.utils.Cuda.init(opt)

  local lm = onmt.lm.LM.new(opt)

  local srcReader = onmt.utils.FileReader.new(opt.src)
  local srcBatch = {}

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
      table.insert(srcBatch, lm:buildInput(srcTokens))
    elseif #srcBatch == 0 then
      break
    end

    if srcTokens == nil or #srcBatch == opt.batch_size then
      if opt.time then
        timer:resume()
      end

      local results
      if opt.mode == 'score' then
        results = lm:evaluate(srcBatch)
      else
        results = lm:sample(srcBatch, opt.max_length, opt.temperature)
      end

      if opt.time then
        timer:stop()
      end

      for b = 1, #results do
        _G.logger:info('SENT %d: %s', sentId, results[b])
        outFile:write(results[b] .. '\n')

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
    _G.logger:info("Average sentence processing time (in seconds):\n")
    _G.logger:info("avg real\t" .. time.real / sentenceCount .. "\n")
    _G.logger:info("avg user\t" .. time.user / sentenceCount .. "\n")
    _G.logger:info("avg sys\t" .. time.sys / sentenceCount .. "\n")
  end

  _G.logger:shutDown()
end

main()
