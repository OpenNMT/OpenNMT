require('onmt.init')

local tester = ...

local tensorTest = torch.TestSuite()

local function reuseTest(tester, a, size)
  local b = onmt.utils.Tensor.reuseTensor(a, size)
  tester:eq(torch.pointer(b:storage()), torch.pointer(a:storage()))
  tester:eq(b:ne(0):sum(), 0)
end

local function compareStorage(a, b, func)
  if torch.isTensor(a) then
    return func(torch.pointer(a:storage()), torch.pointer(b:storage()))
  else
    for k, _ in pairs(a) do
      if not compareStorage(a[k], b[k], func) then
        return false
      end
    end
    return true
  end
end

local function sharedStorage(a, b)
  return compareStorage(a, b, function (p1, p2) return p1 == p2 end)
end
local function nonSharedStorage(a, b)
  return compareStorage(a, b, function (p1, p2) return p1 ~= p2 end)
end


function tensorTest.reuse_smaller()
  local a = torch.Tensor(10, 200)
  reuseTest(tester, a, { 5, 200 })
end

function tensorTest.reuse_same()
  local a = torch.Tensor(10, 200)
  reuseTest(tester, a, a:size())
end

function tensorTest.reuse_multipleResize()
  local a = torch.Tensor(10, 200)
  reuseTest(tester, a, { 5, 200 })
  reuseTest(tester, a, { 10, 200 })
end


function tensorTest.reuseTable()
  local a = {
    torch.Tensor(10, 200),
    torch.Tensor(10, 200)
  }

  local b = onmt.utils.Tensor.reuseTensorTable(a, {5, 100})

  tester:ne(torch.pointer(b), torch.pointer(a))
  tester:eq(sharedStorage(a, b), true)
end


function tensorTest.recursiveApply_empty()
  local identity = function (t)
    return t
  end

  local t = {}
  local expected = {}

  tester:eq(onmt.utils.Tensor.recursiveApply(t, identity), expected)
end

function tensorTest.recursiveApply_inline()
  local addOne = function (t)
    return t:add(1)
  end

  local t = torch.Tensor({0, 0})
  local expected = torch.Tensor({1, 1})

  tester:eq(onmt.utils.Tensor.recursiveApply(t, addOne), expected)
end

function tensorTest.recursiveApply_nested()
  local addOne = function (t)
    return t:add(1)
  end

  local t = {
    {
      torch.Tensor({0, 0}),
      torch.Tensor({0})
    },
    torch.Tensor({0})
  }

  local expected = {
    {
      torch.Tensor({1, 1}),
      torch.Tensor({1})
    },
    torch.Tensor({1})
  }

  tester:eq(onmt.utils.Tensor.recursiveApply(t, addOne), expected)
end


function tensorTest.recursiveClone()
  local t = {
    {
      torch.Tensor({0, 0}),
      torch.Tensor({0})
    },
    torch.Tensor({0})
  }

  local clone = onmt.utils.Tensor.recursiveClone(t)
  tester:eq(clone, t)
  tester:eq(nonSharedStorage(t, clone), true)
end


function tensorTest.deepClone()
  local t = {
    {
      torch.Tensor({0, 0}),
      torch.Tensor({0})
    },
    torch.Tensor({0})
  }

  local clone = onmt.utils.Tensor.deepClone(t)
  tester:eq(clone, t)
  tester:eq(nonSharedStorage(t, clone), true)
end


function tensorTest.recursiveSet()
  local a = {
    {
      torch.Tensor({0, 0}),
      torch.Tensor({0})
    },
    torch.Tensor({0})
  }

  local b = {
    {
      torch.Tensor({1, 1}),
      torch.Tensor({1})
    },
    torch.Tensor({1})
  }

  onmt.utils.Tensor.recursiveSet(a, b)
  tester:eq(a, b)
  tester:eq(sharedStorage(a, b), true)
end


function tensorTest.recursiveAdd_empty()
  local a = {}
  local b = {}
  tester:eq(onmt.utils.Tensor.recursiveAdd(a, b), {})
end

function tensorTest.recursiveAdd_inline()
  local a = torch.Tensor({1, 1})
  local b = torch.Tensor({2, 2})
  tester:eq(onmt.utils.Tensor.recursiveAdd(a, b), torch.Tensor({3, 3}))
end

function tensorTest.recursiveAdd_nested()
  local a = {
    {
      torch.Tensor({1, 1}),
      torch.Tensor({2})
    },
    torch.Tensor({3})
  }
  local b = {
    {
      torch.Tensor({1, 1}),
      torch.Tensor({2})
    },
    torch.Tensor({3})
  }

  local c = {
    {
      torch.Tensor({2, 2}),
      torch.Tensor({4})
    },
    torch.Tensor({6})
  }

  tester:eq(onmt.utils.Tensor.recursiveAdd(a, b), c)
end

return tensorTest
