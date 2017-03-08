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


return {
  subrange = subrange,
  reorder = reorder,
  append = append,
  merge = merge,
  hasValue = hasValue
}
