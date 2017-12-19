#!/usr/bin/env lua
--[[

Author: Vincent Nguyen (Ubiqus)

License MIT

Usage
  Change the section below server_cfg according to your settings
  When querying, "id" : n , where n is the model index in your config.

  th tools/rest_multi_models.lua -model -gpuid 1
  query the server:
  curl -v -H "Content-Type: application/json" -X POST -d '[{ "src" : "international migration" , "id" : 1 }]' http://127.0.0.1:7784/translator/translate


]]

require('onmt.init')

local yaml = require 'yaml'

local tokenizer = require('tools.utils.tokenizer')
local BPE = require ('tools.utils.BPE')
local restserver = require('tools.restserver.restserver')

local cmd = onmt.utils.ExtendedCmdLine.new('rest_multi_models.lua')

local server_options = {
   {
     '-port', 7784,
     [[Port to run the server on.]],
     {
       valid = onmt.utils.ExtendedCmdLine.isUInt
     }
   },
   {
     '-withAttn', false,
     [[If set returns by default attn vector.]]
   },
   {
     '-unload_time', 7,
     [[Unload unused model from memory after this time.]]
   },
   {
     '-model_config', 'tools/rest_config.yml',
     [[Path to yaml configuration file.]],
     {
       valid = onmt.utils.ExtendedCmdLine.fileExists
     }
   }
}

cmd:setCmdLineOptions(server_options, 'Server')
onmt.utils.Cuda.declareOpts(cmd)
onmt.utils.Logger.declareOpts(cmd)

local opt_server = cmd:parse(arg)
local fconfig = io.open(opt_server.model_config, "rb")
assert(fconfig)
local content_config = fconfig:read "*a"
fconfig:close()

local server_cfg = yaml.load(content_config)

-- CAREFULL opt is a table of options here, ie a table of what opt is usually in onmt

local opt = {}

for i=1, #server_cfg do
  local model_cmd = onmt.utils.ExtendedCmdLine.new()
  local model_config = {}
  for k,v in pairs(server_cfg[i]) do
    table.insert(model_config, {'-'..k, v, ''})
  end
  model_cmd:setCmdLineOptions(model_config, 'Server')
  onmt.translate.Translator.declareOpts(model_cmd)
  tokenizer.declareOpts(model_cmd)

  model_cmd:text("")
  model_cmd:text("**Other options**")
  model_cmd:text("")
  model_cmd:option('-batchsize', 1000, [[Size of each parallel batch - you should not change except if low memory.]])
  opt[i] = model_cmd:parse({})
end


local function translateMessage(server, lines)
  local batch = {}
  -- We need to tokenize the input line before translation
  local bpe
  local res
  local err
  -- first item contains both the src (to translate) AND the id of the engine
  -- these local variables are set to match the previous version of code
  local options = server.opt[lines[1].id]
  local translator = server.translator[lines[1].id]

  _G.logger:info("Start Tokenization")
  if options.bpe_model ~= '' then
     bpe = BPE.new(options)
  end
  for i = 1, #lines do
    local srcTokenized = {}
    local tokens
    local srcTokens = {}
    res, err = pcall(function() tokens = tokenizer.tokenize(options, lines[i].src, bpe) end)
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

    for i = 1, translator.args.n_best do
      local srcSent = translator:buildOutput(batch[b])
      local lineres = {
        tgt = "",
        src = srcSent,
        n_best = 1,
        pred_score = 0
      }
      local oline = ""
      if results[b].preds ~= nil then
        local predSent
        res, err = pcall(function()
          predSent = tokenizer.detokenize(options,
                                          results[b].preds[i].words,
                                          results[b].preds[i].features)
        end)
        if not res then
          if string.find(err,"interrupted") then
            error("interrupted")
           else
            error("unicode error in line ".. err)
          end
        end
        lineres = {
          tgt = predSent,
          src = srcSent,
          n_best = i,
          pred_score = results[b].preds[i].score
        }
        if options.withAttn or lines[b].withAttn then
          local attnTable = {}
          for j = 1, #results[b].preds[i].attention do
            table.insert(attnTable, results[b].preds[i].attention[j]:totable())
          end
          lineres.attn = attnTable
        end
      end
      table.insert(ret, lineres)
    end
    table.insert(translations, ret)
  end
  return translations
end

local function init_server(options)
  local server = restserver:new():port(opt_server.port)

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
        if req[1].id == nil then
          req[1].id = 1
        end
        if not server.model_loaded[req[1].id] then
          -- TODO
          -- I need to test here if I have enough memory to load the model
          -- if not then I need to unload the oldest one
          local freeMemory = onmt.utils.Cuda.freeMemory()
          if ( not onmt.utils.Cuda.activated or freeMemory > 3100000000 ) then
            _G.logger:info("Loading model id %d",req[1].id)
            server.translator[req[1].id] = onmt.translate.Translator.new(server.opt[req[1].id])
            server.model_loaded[req[1].id] = true
          else
            return restserver.response():status(500):entity()
          end
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
    if server.elapsed[i] > opt_server.unload_time then
      if server.model_loaded[i] then
        _G.logger:info("unloading model %d",i)
        server.translator[i] = nil
        collectgarbage()
        server.model_loaded[i] = false
      end
    end
    io.stdout:flush()
    io.stderr:flush()
  end
  return false
end


local function main()
  -- load logger
  _G.logger = onmt.utils.Logger.new(opt_server.log_file, opt_server.disable_logs, opt_server.log_level)
  -- cuda settings
  onmt.utils.Cuda.init(opt_server)
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
  server:enable("tools.restserver.restserver.xavante"):start(function() is_finished(server); end, 3)
end

main()


