-- tds is lazy loaded.
local tds

--[[ Return subset of table ]]
local function subrange(t, first, count)
  local sub = {}
  for i=first,first+count-1 do
    sub[#sub + 1] = t[i]
  end
  return sub
end

--[[ Append table `src` to `dst`. ]]
local function append(dst, src)
  for i = 1, #src do
    table.insert(dst, src[i])
  end
end

--[[ Merge dict `src` to `dst`. ]]
local function merge(dst, src)
  for k, v in pairs(src) do
    dst[k] = v
  end
end

local function empty (self)
  if next(self) == nil then
    return true
  else
    return false
  end
end

--[[ Reorder table `tab` based on the `index` array. ]]
local function reorder(tab, index, cdata)
  local newTab
  if cdata then
    if not tds then
      tds = require('tds')
    end
    newTab = tds.Vec()
    newTab:resize(#tab)
  else
    newTab = {}
  end

  for i = 1, #tab do
    newTab[i] = tab[index[i]]
  end

  return newTab
end

--[[ Check if value is part of list/table. ]]
local function hasValue(tab, value)
  for _, v in ipairs(tab) do
    if v == value then
      return true
    end
  end
  return false
end

local function deepCopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[deepCopy(orig_key)] = deepCopy(orig_value)
        end
        setmetatable(copy, deepCopy(getmetatable(orig)))
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end

return {
  subrange = subrange,
  reorder = reorder,
  append = append,
  merge = merge,
  hasValue = hasValue,
  empty = empty,
  deepCopy = deepCopy
}
