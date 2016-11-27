--[[
   This file provides generic parallel class - allowing to run functions
   in different threads and on different GPU
]]--

local cuda = require 'lib.utils.cuda'

local Parallel = {
   gpus = {0},
   _pool = nil,
   count = 1
}

function Parallel.init(args)
   if cuda.activated then
      Parallel.count = args.nparallel
      Parallel.gpus = cuda.getGPUs(args.nparallel)
      if Parallel.count > 1 then
         local threads = require 'threads'
         threads.Threads.serialization('threads.sharedserialize')
         local thegpus = Parallel.gpus
         Parallel._pool = threads.Threads(
            Parallel.count,
            function(threadid)
               require 'cunn'
               require 'nngraph'
               require('lib.utils.init')
               require('lib.train.init')
               require('lib.onmt.init')
               Data = require('lib.data')
               print('STARTING THREAD ', threadid, 'on GPU ' .. thegpus[threadid])
               utils.Cuda.init(args, thegpus[threadid])
               local refprint = print
               print=function(...) refprint('[THREAD'..threadid..']',...) end
            end
         ) -- dedicate threads to GPUs
         Parallel._pool:specific(true)
      else
         Parallel._G = {}
         Parallel._G.Decoder = require 's2sa.decoder'
         Parallel._G.Encoder = require 's2sa.encoder'
         Parallel._G.Generator = require 's2sa.generator'
      end
   end
end

function Parallel.getGPU(i)
   if cuda.activated and Parallel.gpus[i] ~= 0 then
      return Parallel.gpus[i]
   end
   return 0
end

function Parallel.launch(label, closure, endcallback)
   endcallback = endcallback or function() end
   if label ~= nil then
      print("START",label)
   end
   for j = 1, Parallel.count do
      if Parallel._pool == nil then
         local _G = Parallel._G
         endcallback(closure(j))
      else
         Parallel._pool:addjob(j, function() return closure(j) end, endcallback)
      end
   end
   if Parallel._pool then
      Parallel._pool:synchronize()
   end
   if label ~= nil then
      print("DONE",label)
   end
end

return Parallel
