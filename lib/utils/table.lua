--[[ Append table `src` to `dst`. ]]
local function append(dst, src)
  for i = 1, #src do
    table.insert(dst, src[i])
  end
end

--[[ Reorder table `tab` based on the `index` array. ]]
local function reorder(tab, index)
  local new_tab = {}
  for i = 1, #tab do
    table.insert(new_tab, tab[index[i]])
  end
  return new_tab
end

--[[ Clone table `tab` using `clone()` on each element. ]]
local function clone(tab)
  local new_tab = {}
  for i = 1, #tab do
    table.insert(new_tab, tab[i]:clone())
  end
  return new_tab
end

return {
  clone = clone,
  reorder = reorder,
  append = append
}
