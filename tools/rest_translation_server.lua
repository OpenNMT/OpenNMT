#!/usr/bin/env lua
--[[
  This requires the restserver-xavante rock to run.
  run server (this file)
  th tools/rest_translation_server.lua -model ../Recipes/baseline-1M-enfr/exp/model-baseline-1M-enfr_epoch13_3.44.t7 -gpuid 1
  query the server:
  curl -v -H "Content-Type: application/json" -X POST -d '[{ "src" : "international migration" }]' http://127.0.0.1:7784/translator/translate
]]

require('onmt.init')
local tokenizer = require('tools.utils.tokenizer')
local BPE = require ('tools.utils.BPE')
local restserver = require("tools.restserver.restserver")

local cmd = onmt.utils.ExtendedCmdLine.new('rest_translation_server.lua')

local options = {
  {
    '-host', '127.0.0.1',
    [[Host to run the server on.]]
  },
  {
    '-port', '7784',
    [[Port to run the server on.]]
  },
  {
    '-withAttn', false,
    [[If set returns by default attn vector.]]
  }
}

cmd:setCmdLineOptions(options, 'Server')

onmt.translate.Translator.declareOpts(cmd)

onmt.utils.Cuda.declareOpts(cmd)
onmt.utils.Logger.declareOpts(cmd)
tokenizer.declareOpts(cmd)
onmt.utils.HookManager.updateOpt(arg, cmd)
onmt.utils.HookManager.declareOpts(cmd)

cmd:text("")
cmd:text("Other options")
cmd:text("")

cmd:option('-batchsize', 1000, [[Size of each parallel batch - you should not change except if low memory.]])

local opt = cmd:parse(arg)

local function translateMessage(translator, lines)
  local batch = {}
 -- We need to tokenize the input line before translation
  local bpe
  local res
  local err
  _G.logger:debug("Start Tokenization")
  if opt.bpe_model ~= '' then
    bpe = BPE.new(opt)
  end
  for i = 1, #lines do
    local srcTokenized = {}
    local tokens
    local srcTokens = {}
    res, err = pcall(function()
      local preprocessed = _G.hookManager:call("mpreprocess", opt, lines[i].src) or lines[i]
      tokens = tokenizer.tokenize(opt, preprocessed, bpe)
    end)
     -- it can generate an exception if there are utf-8 issues in the text
    if not res then
      if string.find(err, "interrupted") then
        error("interrupted")
      else
        error("unicode error in line " .. err)
      end
    end
    table.insert(srcTokenized, table.concat(tokens, ' '))
    -- Extract from the line.
    for word in srcTokenized[1]:gmatch'([^%s]+)' do
      table.insert(srcTokens, word)
    end
    -- Currently just a single batch.
    table.insert(batch, translator:buildInput(srcTokens))
  end
  -- Translate
  _G.logger:debug("Start Translation")
  local results = translator:translate(batch)
  _G.logger:debug("End Translation")

  -- Return the nbest translations for each in the batch.
  local translations = {}
  for b = 1, #lines do
    local ret = {}
    for i = 1, translator.args.n_best do
      local srcSent = translator:buildOutput(batch[b])
      local predSent = translator:buildOutput(results[b].preds[i])

      local oline
      res, err = pcall(function() oline = tokenizer.detokenize(predSent, opt) end)
      if not res then
        if string.find(err,"interrupted") then
          error("interrupted")
         else
          error("unicode error in line ".. err)
        end
      end

      local lineres = {
        tgt = oline,
        src = srcSent,
        n_best = i,
        pred_score = results[b].preds[i].score
      }
      if opt.withAttn or lines[b].withAttn then
        local attnTable = {}
        for j = 1, #results[b].preds[i].attention do
          table.insert(attnTable, results[b].preds[i].attention[j]:totable())
        end
        lineres.attn = attnTable
      end
      table.insert(ret, lineres)
    end
    table.insert(translations, ret)
  end
  return translations
end

local function init_server(host, port, translator)
  local server = restserver:new():host(host):port(port)

  server:add_resource("translator", {
    {
      method = "POST",
      path = "/translate",
      consumes = "application/json",
      produces = "application/json",
      handler = function(req)
        _G.logger:debug("receiving request")
        local translate = translateMessage(translator, req)
        _G.logger:debug("sending response")
        return restserver.response():status(200):entity(translate)
      end,
    }
  })
  return server
end

local function main()
  -- load logger
  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)
  onmt.utils.Cuda.init(opt)

  _G.hookManager = onmt.utils.HookManager.new(opt)

  -- disable profiling
  _G.profiler = onmt.utils.Profiler.new(false)

  _G.logger:info("Loading model")
  local translator = onmt.translate.Translator.new(opt)
  _G.logger:info("Launch server")
  local server = init_server(opt.host, opt.port, translator)
  -- This loads the restserver.xavante plugin
  server:enable("restserver.xavante"):start()
end

main()

