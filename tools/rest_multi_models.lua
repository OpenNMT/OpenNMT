#!/usr/bin/env lua
--[[
  This requires the restserver-xavante rock to run.
  run server (this file)
  th tools/rest_multi_models.lua -model -gpuid 1
  query the server:
  curl -v -H "Content-Type: application/json" -X POST -d '[{ "src" : "international migration" , "id" : 1 }]' http://127.0.0.1:7784/translator/translate
]]

require('onmt.init')

local tokenizer = require('tools.utils.tokenizer')
local BPE = require ('tools.utils.BPE')
local restserver = require('tools.restserver.restserver')

--[[
  The following should be in a YAML file instead.
  First item contains the port of the REST server
]]

local server_cfg = {
   {
     {'-model', '/nmt/NMTModels/fr-en/model-fren_epoch10_4.10.t7', [[Model path]]},
     {'-replace_unk' , true, [[If set replace unk words]]},
     {'-port', '7784', [[Port to run the server on.]]},
     {'-withAttn', false, [[If set returns by default attn vector.]]}
   }
   ,
   {
     {'-model', '/nmt/NMTModels/en-fr/model-enfr_epoch10_3.42.t7', [[Model path]]},
     {'-replace_unk' , true, [[If set replace unk words]]},
     {'-withAttn', false, [[If set returns by default attn vector.]]}
   }
}

-- CAREFULL opt is a table of options here, ie a table of what opt is usually in onmt

local cmd = {}
local opt = {}

for i=1, #server_cfg do
  cmd[i] = onmt.utils.ExtendedCmdLine.new('rest_translation_server.lua')
  cmd[i]:setCmdLineOptions(server_cfg[i], 'Server')
  onmt.translate.Translator.declareOpts(cmd[i])
  onmt.utils.Cuda.declareOpts(cmd[i])
  onmt.utils.Logger.declareOpts(cmd[i])
  tokenizer.declareOpts(cmd[i])

  cmd[i]:text("")
  cmd[i]:text("**Other options**")
  cmd[i]:text("")
  cmd[i]:option('-batchsize', 1000, [[Size of each parallel batch - you should not change except if low memory.]])

  opt[i] = cmd[i]:parse(arg)
end


local function translateMessage(server,lines)
  local batch = {}
 -- We need to tokenize the input line before translation
  local bpe
  local res
  local err
 -- first item contains both the src (to translate) AND the id of the engine
 -- these local variables are set to match the previous version of code
  local opt = server.opt[lines[1].id]
  local translator = server.translator[lines[1].id]

  _G.logger:info("Start Tokenization")
  if opt.bpe_model ~= '' then
     bpe = BPE.new(opt)
  end
  for i = 1, #lines do
    local srcTokenized = {}
    local tokens
    local srcTokens = {}
    res, err = pcall(function() tokens = tokenizer.tokenize(opt, lines[i].src, bpe) end)
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
  _G.logger:info("Start Translation")
  local results = translator:translate(batch)
  _G.logger:info("End Translation")

  -- Return the nbest translations for each in the batch.
  local translations = {}
  for b = 1, #lines do
    local ret = {}
    for i = 1, translator.opt.n_best do
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

local function init_server(options)
  local server = restserver:new(options):port(options[1].port)

  server.opt = options
  server:add_resource("translator", {
    {
      method = "POST",
      path = "/translate",
      consumes = "application/json",
      produces = "application/json",
      handler = function(req)
        _G.logger:info("receiving request for model id %d",req[1].id)
 -- remember the first item contains also the id of the engine
 -- for backward compatibility maybe if req[1].id not defined, set it to 1
        if req[1].id = nil then
           req[1].id = 1
        end
        if not server.model_loaded[req[1].id] then
 -- I need to test here I f I have enough memory to load the model
 -- if not then I need to unload the oldest one
 -- print(cutorch.getMemoryUsage())
          _G.logger:info("Loading model id %d",req[1].id)
          server.translator[req[1].id] = onmt.translate.Translator.new(server.opt[req[1].id])
          server.model_loaded[req[1].id] = true
        end  
        server.timer[req[1].id] = torch.Timer()
        local translate = translateMessage(server,req)
        _G.logger:info("sending response model id %d",req[1].id)
        return restserver.response():status(200):entity(translate)
      end,
    }
  })
  return server
end


local function is_finished(server)
   for i=1, #server_cfg do
     server.elapsed[i] = server.timer[i]:time().real
--     print("Model %d %s time elapsed %d",i,server.model_loaded[i],server.elapsed[i])
     if server.elapsed[i] > 10 then
       if server.model_loaded[i] then
         print("unloading model %d",i)
         server.translator[i] = nil
         collectgarbage()
         server.model_loaded[i] = false
       end
     end
   end
  return false
end


local function main()
-- load logger
  _G.logger = onmt.utils.Logger.new(nil, false, 'INFO')

  onmt.utils.Cuda.init(opt[1])

  -- disable profiling
  _G.profiler = onmt.utils.Profiler.new(false)

  _G.logger:info("Launch server")
  local server = init_server(opt)

   for i=1, #server_cfg do
      server.timer[i] = torch.Timer()
      server.elapsed[i] = 0
      server.model_loaded[i] = false
      server.translator[i] = nil
   end

  -- This loads the restserver.xavante plugin
  server:enable("tools.restserver.restserver.xavante"):start(function() is_finished(server); end,3)
 
end

main()

