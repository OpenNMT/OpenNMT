local threads = require('threads')

local ThreadPool = {
  numInstances = 1
}

function ThreadPool.init(numInstances, ...)
  assert(numInstances > 0)

  ThreadPool.numInstances = numInstances

  if numInstances > 1 then
    threads.Threads.serialization('threads.sharedserialize')
    ThreadPool._pool = threads.Threads(numInstances, ...)
    ThreadPool._pool:specific(true)
  end
end

function ThreadPool.dispatch(fun, callback)
  callback = callback or function () end

  for i = 1, ThreadPool.numInstances do
    if not ThreadPool._pool then
      callback(fun(i))
    else
      ThreadPool._pool:addjob(i, function () return fun(i) end, callback)
    end
  end

  if ThreadPool._pool then
    ThreadPool._pool:synchronize()
  end
end

return ThreadPool
