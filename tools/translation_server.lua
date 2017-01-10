local zmq = require("zmq")
local json = require("json")

require('onmt.init')

local cmd = torch.CmdLine()

cmd:text("")
cmd:text("**onmt.translation_server**")
cmd:text("")

cmd:option('-config', '', [[Read options from this file]])
onmt.translate.Translator.declareOpts(cmd)

cmd:option('-host', '127.0.0.1', [[Host to run the server on]])
cmd:option('-port', '5556', [[Port to run the server on]])
cmd:text("")
cmd:text("**Other options**")
cmd:text("")
cmd:option('-gpuid', -1, [[ID of the GPU to use (-1 = use CPU, 0 = let cuda choose between available GPUs)]])
cmd:option('-fallback_to_cpu', false, [[If = true, fallback to CPU if no GPU available]])
cmd:option('-log_file', '', [[Outputs logs to a file under this path instead of stdout.]])
cmd:option('-disable_logs', false, [[If = true, output nothing.]])


local function translateMessage(translator, lines)
  local srcBatch = {}
  local srcWordsBatch = {}
  local srcFeaturesBatch = {}

  -- Extract from the line.
  for i = 1, #lines do
    local srcTokens = {}
    for word in lines[i].src:gmatch'([^%s]+)' do
      table.insert(srcTokens, word)
    end
    local srcWords, srcFeats = onmt.utils.Features.extract(srcTokens)

    -- Currently just a single batch.
    table.insert(srcBatch, srcTokens)
    table.insert(srcWordsBatch, srcWords)
    if #srcFeats > 0 then
      table.insert(srcFeaturesBatch, srcFeats)
    end
  end

  -- Translate
  local data = translator:buildData(srcWordsBatch, srcFeaturesBatch,
                                    nil, nil)
  local batch = data:getBatch()
  local pred, predFeats, predScore, attn = translator:translateBatch(batch)

  -- Return the nbest translations for each in the batch.
  local translations = {}
  for b = 1, #lines do
    local ret = {}
    for i = 1, translator.opt.n_best do
      local predBatch = translator:buildTargetTokens(pred[b][i], predFeats[b][i],
                                                     srcBatch[b], attn[b][i])
      local predSent = predBatch
      local attnTable = {}
      for j = 1, #attn[b][i] do
        table.insert(attnTable, attn[b][i][j]:totable())
      end
      local srcSent = srcBatch[b]
      table.insert(ret, {tgt = predSent, attn = attnTable, src=srcSent, n_best=i,
                         pred_score=predScore[b][i]})
    end
    table.insert(translations, ret)
  end
  return translations
end

local function main()
  local opt = cmd:parse(arg)
  local requiredOptions = {
    "model"
  }

  onmt.utils.Opt.init(opt, requiredOptions)

  local mute = (opt.log_file:len() > 0)
  _G.logger = onmt.utils.Logger.new(opt.log_file, mute)
  if opt.disable_logs then
    _G.logger:setVisibleLevel('ERROR')
  end

  _G.logger:info("Loading model")
  local translator = onmt.translate.Translator.new(opt)

  local ctx = zmq.init(1)
  local s = ctx:socket(zmq.REP)

  local url = "tcp://" .. opt.host .. ":" .. opt.port
  s:bind(url)
  _G.logger:info("Server initialized at " .. url)
  while true do
    -- Input format is a json batch of src strings.
    local recv = s:recv()
    _G.logger:info("Received... " .. recv)
    local message = json.decode(recv)

    local translate = translateMessage(translator, message)
    local ret = json.encode(translate)
    s:send(ret)
    _G.logger:info("Returning... " .. ret)
    collectgarbage()
  end
  _G.logger:shutDown()
end

main()
