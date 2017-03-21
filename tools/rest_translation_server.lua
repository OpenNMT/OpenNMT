#!/usr/bin/env lua
--[[
  This requires the restserver-xavante rock to run.
  run server (this file)
  th tools/rest_translation_server.lua -model ../Recipes/baseline-1M-enfr/exp/model-baseline-1M-enfr_epoch13_3.44.t7 -gpuid 1
  query the server:
  curl -v -H "Content-Type: application/json" -X POST -d '{ "src" : "international migration" }' http://127.0.0.1:7784/translator/translate
]]

require('onmt.init')

local separators = require('tools.utils.separators')
local tokenizer = require('tools.utils.tokenizer')
local BPE = require ('tools.utils.BPE')
local restserver = require("restserver")

local cmd = onmt.utils.ExtendedCmdLine.new('rest_translation_server.lua')

local options = {
   {'-port', '7784', [[Port to run the server on.]]},
   {'-withAttn', false, [[If set returns by default attn vector.]]}
}

cmd:setCmdLineOptions(options, 'Server')

onmt.translate.Translator.declareOpts(cmd)

cmd:text("")
cmd:text("**Other options**")
cmd:text("")
onmt.utils.Cuda.declareOpts(cmd)
onmt.utils.Logger.declareOpts(cmd)

cmd:option('-mode', 'conservative', [[Define how aggressive should the tokenization be - 'aggressive'
  only keeps sequences of letters/numbers, 'conservative' allows mix of alphanumeric as in: '2,000', 'E65', 'soft-landing'.]])
cmd:option('-joiner_annotate', false, [[Include joiner annotation using 'joiner' character.]])
cmd:option('-joiner', separators.joiner_marker, [[Character used to annotate joiners.]])
cmd:option('-joiner_new', false, [[in joiner_annotate mode, 'joiner' is an independent token.]])
cmd:option('-case_feature', false, [[Generate case feature.]])
cmd:option('-bpe_model', '', [[Apply Byte Pair Encoding if the BPE model path is given.]])
cmd:option('-batchsize', 1000, [[Size of each parallel batch - you should not change except if low memory.]])

local opt = cmd:parse(arg)

local function translateMessage(translator, req)
  local batch = {}
  -- We need to tokenize the input line before translation
  local srcTokens = {}
  local bpe
  local srcTokenized = {}
  local res
  local err
  local tokens
  _G.logger:info("Start Tokenization")
  if opt.bpe_model ~= '' then
     bpe = BPE.new(opt.bpe_model, opt.joiner_annotate, opt.joiner_new)
  end
  res, err = pcall(function() tokens = tokenizer.tokenize(opt, req.src, bpe) end)
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

  _G.logger:info("Start Translation")
  -- Currently just a single batch.
  table.insert(batch, translator:buildInput(srcTokens))
  -- Translate
  local results = translator:translate(batch)
  _G.logger:info("End Translation")

  -- Return the nbest translations for each in the batch.
  local translations = {}
  local ret = {}
  for i = 1, translator.opt.n_best do
    local srcSent = translator:buildOutput(batch[1])
    local predSent = translator:buildOutput(results[1].preds[i])

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
      pred_score = results[1].preds[i].score
    }
    if opt.withAttn or req.withAttn then
      local attnTable = {}
      for j = 1, #results[1].preds[i].attention do
        table.insert(attnTable, results[1].preds[i].attention[j]:totable())
      end
      lineres.attn = attnTable
    end
    table.insert(ret, lineres)
  end
  table.insert(translations, ret)

  return translations
end

local function init_server(port, translator)
  local server = restserver:new():port(port)

  server:add_resource("translator", {
    {
      method = "POST",
      path = "/translate",
      consumes = "application/json",
      produces = "application/json",
      handler = function(req)
        _G.logger:info("receiving request: [%s]", req.src:gsub("\n", "\\n"))
        local translate = translateMessage(translator, req)
        _G.logger:info("sending response")
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

  -- disable profiling
  _G.profiler = onmt.utils.Profiler.new(false)

  _G.logger:info("Loading model")
  local translator = onmt.translate.Translator.new(opt)
  _G.logger:info("Launch server")
  local server = init_server(opt.port, translator)
  -- This loads the restserver.xavante plugin
  server:enable("restserver.xavante"):start()
end

main()

