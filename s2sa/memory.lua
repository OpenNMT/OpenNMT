-- module for memory management

-- reuseMem is used for reusing output tensor for storing gradInput and optimizing memory allocation
-- use :reuseMem() on the module to allow the feature
-- then apply setReuse after initialization
-- only applies if output and gradinput are of the same type
function nn.Module:reuseMem()
  self.reuse = true
  return self
end

function nn.Module:setReuse()
  if self.reuse then
    assert(type(self.output) == type(self.gradInput), "invalid use of reuseMem")
    self.gradInput = self.output
  end
  return self
end

-- usePrealloc is based on the same principle but use pre-allocated memory at the beginning of the process that can be shared
-- between different objects
-- use to prellocate gradInput, or output - useful for intermediate calculations working on large input
preallocWarning = {}
preallocTable = {}

function nn.Module:usePrealloc(preallocName)
  self.prealloc = preallocName
  return self
end

function nn.Module:setPrealloc()
  if self.prealloc then
    if preallocTable[self.prealloc] == nil then
      if not(preallocWarning[self.prealloc]) then
        print('WARNING: no prealloc memory defined for \'' .. self.prealloc .. '\'')
        preallocWarning[self.prealloc] = 1
      end
      return
    end
    local memmap = preallocTable[self.prealloc]
    if memmap["GI"] ~= nil then
      assert(type(self.gradInput) == type(memmap.GI), "invalid use of usePrealloc")
      self.gradInput = memmap["GI"]
    end
    if memmap["O"] ~= nil then
      assert(type(self.output) == type(memmap.O), "invalid use of usePrealloc")
      self.output = memmap["O"]
    end
  end
  return self
end
