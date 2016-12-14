-- taken from https://github.com/harvardnlp/seq2seq-attn/blob/master/s2sa/memory.lua
-- module for memory management

-- reuseMem is used for reusing output tensor for storing gradInput and optimizing memory allocation
-- use :reuseMem() on the module to allow the feature
-- then apply setReuse after initialization
-- only applies if output and gradinput are of the same type
require 'utils'

if logging ~= nil then
    log = function(msg) logging:info(msg) end
else
    log = print
end
function nn.Module:reuseMem(name)
  self.reuse = true
  return self
end

function nn.Module:setReuse()
  if self.reuse then
    assert(type(self.output) == type(self.gradInput), "invalid use of reuseMem:")
    self.gradInput = self.output
  end
  return self
end

-- usePrealloc is based on the same principle but use pre-allocated memory at the beginning of the process that can be shared
-- between different objects
-- use to prellocate gradInput, or output - useful for intermediate calculations working on large input
preallocTable = nil

function preallocateMemory(switch)
  if switch then
    preallocTable = {}
    log('Switching on memory preallocation')
  end
end

function preallocateTensor(name,D)
  if #D > 1 then
    local T={}
    for i=1,#D do
      table.insert(T,preallocateTensor(name,{D[i]}))
    end
    return T
  else
    D = D[1]
  end
  local t=localize(torch.zeros(torch.LongStorage(D)))
  return t
end

-- enable reuseMemory - if preallocation disable, then switched back to reuseMem checking for 'reuse' in name
function nn.Module:usePrealloc(preallocName, inputDim, outputDim)
  if preallocTable == nil then
    if string.find(preallocName, "reuse") then
      self:reuseMem()
    end
    return self
  end
  self.prealloc = preallocName
  self.name = preallocName
  self.preallocInputDim = inputDim
  self.preallocOutputDim = outputDim
  return self
end

function nn.Module:setPrealloc()
  if self.prealloc and (self.preallocInputDim ~= nil or self.preallocOutputDim ~= nil) then
    if preallocTable[self.prealloc] == nil then
      preallocTable[self.prealloc] = {
      }
      if self.preallocInputDim ~= nil then
        preallocTable[self.prealloc].GI = preallocateTensor(self.prealloc, self.preallocInputDim)
      end
      if self.preallocOutputDim ~= nil then
        preallocTable[self.prealloc].O = preallocateTensor(self.prealloc, self.preallocOutputDim)
      end
    end
    local memmap = preallocTable[self.prealloc]
    if memmap["GI"] ~= nil then
      assert(type(self.gradInput) == type(memmap.GI), "invalid use of usePrealloc ["..self.prealloc.."]/GI: "..type(self.gradInput).."/"..type(memmap.GI))
      self.gradInput = memmap["GI"]
    end
    if memmap["O"] ~= nil then
      assert(type(self.output) == type(memmap.O), "invalid use of usePrealloc ["..self.prealloc.."]/O:"..type(self.output).."/"..type(memmap.O))
      self.output = memmap["O"]
    end
  end
  return self
end
