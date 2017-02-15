#!/usr/bin/env lua
--[[
This requires the restserver-xavante rock to run.
run server (this file)
th tools/rest_translation_server.lua -model ../Recipes/baseline-1M-enfr/exp/model-baseline-1M-enfr_epoch13_3.44.t7 -gpuid 1
query the server
curl -v -H "Content-Type: application/json" -X POST -d '{ "src" : "international migration" }' http://127.0.0.1:8080/translator/translate
]]

require('onmt.init')

local threads = require 'threads'
local separators = require('tools.utils.separators')
local tokenizer = require('tools.utils.tokenizer')

local cmd = torch.CmdLine()

cmd:text("")
cmd:text("**onmt.rest_translation_server**")
cmd:text("")

cmd:option('-config', '', [[Read options from this file]])
onmt.translate.Translator.declareOpts(cmd)

cmd:option('-host', '127.0.0.1', [[Host to run the server on]])
cmd:option('-port', '8080', [[Port to run the server on]])
cmd:option('-gpuid', 1, [[]])
cmd:text("")
cmd:text("**Other options**")
cmd:text("")
onmt.utils.Cuda.declareOpts(cmd)
onmt.utils.Logger.declareOpts(cmd)

cmd:option('-mode', 'conservative', [[Define how aggressive should the tokenization be - 'aggressive' only keeps sequences of letters/numbers,
  'conservative' allows mix of alphanumeric as in: '2,000', 'E65', 'soft-landing']])
cmd:option('-joiner_annotate', true, [[Include joiner annotation using 'joiner' character]])
cmd:option('-joiner', separators.joiner_marker, [[Character used to annotate joiners]])
cmd:option('-joiner_new', false, [[in joiner_annotate mode, 'joiner' is an independent token]])
cmd:option('-case_feature', true, [[Generate case feature]])
cmd:option('-bpe_model', '', [[Apply Byte Pair Encoding if the BPE model path is given]])
cmd:option('-nparallel', 1, [[Number of parallel thread to run the tokenization]])
cmd:option('-batchsize', 1000, [[Size of each parallel batch - you should not change except if low memory]])


local function translateMessage(translator, lines)
  local batch = {}
  -- We need to tokenize the input line before translation
    local srcTokens = {}
    local bpe
    local srcTokenized = {}
    local res
    local err
    local tokens
    BPE = require ('tools.utils.BPE')
    if opt.bpe_model ~= '' then
       bpe = BPE.new(opt.bpe_model, opt.joiner_annotate, opt.joiner_new)
    res, err = pcall(function() tokens = tokenizer.tokenize(opt, lines.src, bpe) end)
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

  -- Translate
  local results = translator:translate(batch)

  -- Return the nbest translations for each in the batch.
  local translations = {}
    local ret = {}
    for i = 1, translator.opt.n_best do
      local srcSent = translator:buildOutput(batch[1])
      local predSent = translator:buildOutput(results[1].preds[i])

      local oline
      res, err = pcall(function() oline = tokenizer.detokenize(predSent, opt) end)
--    table.insert(output, oline)
      if not res then
        if string.find(err,"interrupted") then
          error("interrupted")
         else
          error("unicode error in line ".. err)
        end
      end

      local attnTable = {}
      for j = 1, #results[1].preds[i].attention do
        table.insert(attnTable, results[1].preds[i].attention[j]:totable())
      end

      table.insert(ret, {
        tgt = oline,
        attn = attnTable,
        src = srcSent,
        n_best = i,
        pred_score = results[1].preds[i].score
      })
    end
    table.insert(translations, ret)

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
  translator = onmt.translate.Translator.new(opt)

  -- This loads the restserver.xavante plugin
  server:enable("restserver.xavante"):start()
end

local restserver = require("restserver")

local server = restserver:new():port(opt.port)

server:add_resource("translator", {
   {
      method = "POST",
      path = "/translate",
      consumes = "application/json",
      produces = "application/json",
      input_schema = {
         src = { type = "string" },
      },
      handler = function(req)
         print("receiving request: ")
         print(req)
         local translate = translateMessage(translator, req)
         print("sending response: ")
         print(translate)
         return restserver.response():status(200):entity(translate)
      end,
   },
})

main()



