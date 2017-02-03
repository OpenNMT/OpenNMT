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
onmt.utils.Cuda.declareOpts(cmd)
onmt.utils.Logger.declareOpts(cmd)

local function extractData(tokens)
  local words, features = onmt.utils.Features.extract(tokens)

  local data = {}
  data.words = words

  if #features > 0 then
    data.features = features
  end

  return data
end

local function buildSentence(data)
  return table.concat(onmt.utils.Features.annotate(data.words, data.features), " ")
end

local function translateMessage(translator, lines)
  local batch = {}

  -- Extract from the line.
  for i = 1, #lines do
    local srcTokens = {}
    for word in lines[i].src:gmatch'([^%s]+)' do
      table.insert(srcTokens, word)
    end

    -- Currently just a single batch.
    table.insert(batch, extractData(srcTokens))
  end

  -- Translate
  local results = translator:translate(batch)

  -- Return the nbest translations for each in the batch.
  local translations = {}

  for b = 1, #lines do
    local ret = {}

    for i = 1, translator.opt.n_best do
      local srcSent = buildSentence(batch[b])
      local predSent = buildSentence(results[b].preds[i])

      local attnTable = {}
      for j = 1, #results[b].preds[i].attention do
        table.insert(attnTable, results[b].preds[i].attention[j]:totable())
      end

      table.insert(ret, {
        tgt = predSent,
        attn = attnTable,
        src = srcSent,
        n_best = i,
        pred_score = results[b].preds[i].score
      })
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

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)

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

    local ret
    local _, err = pcall(function ()
      local translate = translateMessage(translator, message)
      ret = json.encode(translate)
    end)

    if err then
      -- Hide paths included in the error message.
      err = err:gsub("/[^:]*/", "")
      ret = json.encode({ error = err })
    end

    s:send(ret)
    _G.logger:info("Returning... " .. ret)
    collectgarbage()
  end
end

main()
