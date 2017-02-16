--[[ Recursively call `func()` on all tensors within `out`. ]]
local function recursiveApply(out, func, ...)
  local res
  if torch.type(out) == 'table' then
    res = {}
    for k, v in pairs(out) do
      res[k] = recursiveApply(v, func, ...)
    end
    return res
  end
  if torch.isTensor(out) then
    res = func(out, ...)
  else
    res = out
  end
  return res
end

--[[ Recursively call `clone()` on all tensors within `out`. ]]
local function recursiveClone(out)
  return recursiveApply(out, function (h) return h:clone() end)
end

--[[ Recursively add `b` tensors into `a`'s. ]]
local function recursiveAdd(a, b)
  if torch.isTensor(a) then
    a:add(b)
  else
    for i = 1, #a do
      recursiveAdd(a[i], b[i])
    end
  end
  return a
end

local function recursiveSet(dst, src)
  if torch.isTensor(dst) then
    dst:set(src)
  else
    for k, _ in ipairs(dst) do
      recursiveSet(dst[k], src[k])
    end
  end
end

--[[ Clone any serializable Torch object. ]]
local function deepClone(obj)
  local mem = torch.MemoryFile("rw"):binary()
  mem:writeObject(obj)
  mem:seek(1)
  local clone = mem:readObject()
  mem:close()
  return clone
end

--[[
Reuse Tensor storage and avoid new allocation unless any dimension
has a larger size.

Parameters:

  * `t` - the tensor to be reused
  * `sizes` - a table or tensor of new sizes

Returns: a view on zero-tensor `t`.

--]]
local function reuseTensor(t, sizes)
  assert(t ~= nil, 'tensor must not be nil for it to be reused')

  if torch.type(sizes) == 'table' then
    sizes = torch.LongStorage(sizes)
  end

  return t:resize(sizes):zero()
end

--[[
Reuse all Tensors within the table with new sizes.

Parameters:

  * `tab` - the table of tensors
  * `sizes` - a table of new sizes

Returns: a table of tensors using the same storage as `tab`.

--]]
local function reuseTensorTable(tab, sizes)
  local newTab = {}

  for i = 1, #tab do
    table.insert(newTab, reuseTensor(tab[i], sizes))
  end

  return newTab
end

--[[
Initialize a table of tensors with the given sizes.

Parameters:

  * `size` - the number of clones to create
  * `proto` - tensor to be clone for each index
  * `sizes` - a table of new sizes

Returns: an initialized table of tensors.

--]]
local function initTensorTable(size, proto, sizes)
  local tab = {}

  local base = reuseTensor(proto, sizes)

  for _ = 1, size do
    table.insert(tab, base:clone())
  end

  return tab
end

--[[
Copy tensors from `src` reusing all tensors from `proto`.

Parameters:

  * `proto` - the table of tensors to be reused
  * `src` - the source table of tensors

Returns: a copy of `src`.

--]]
local function copyTensorTable(proto, src)
  local tab = reuseTensorTable(proto, src[1]:size())

  for i = 1, #tab do
    tab[i]:copy(src[i])
  end

  return tab
end

return {
  recursiveApply = recursiveApply,
  recursiveClone = recursiveClone,
  recursiveAdd = recursiveAdd,
  recursiveSet = recursiveSet,
  deepClone = deepClone,
  reuseTensor = reuseTensor,
  reuseTensorTable = reuseTensorTable,
  initTensorTable = initTensorTable,
  copyTensorTable = copyTensorTable
}
