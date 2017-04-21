require('onmt.init')

local tester = ...

local profileTest = torch.TestSuite()

function profileTest.profiling()
  local profiler = onmt.utils.Profiler.new({profiler=true})
  profiler:start("main")
  local count = 0
  while count < 100 do count = count+1 end
  profiler:start("a")
  while count < 1000 do count = count+1 end
  profiler:stop("a"):start("b.c")
  while count < 10000 do count = count+1 end
  profiler:stop("b.c"):start("b.d"):stop("b.d")
  profiler:stop("main")
  local v=profiler:log():gsub("[-0-9.e]+","*")
  tester:assert(v=="main:{total:*,a:*,b:{total:*,d:*,c:*}}" or v == "main:{total:*,a:*,b:{total:*,c:*,d:*}}"
                or v == "main:{total:*,b:{total:*,c:*,d:*},a:*}" or v == "main:{total:*,b:{total:*,d:*,c:*},a:*}")
end

return profileTest
