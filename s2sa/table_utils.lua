local function zero(t)
  for i = 1, #t do
    t[i]:zero()
  end
end

local function append(dst, src)
  for i = 1, #src do
    table.insert(dst, src[i])
  end
end

local function reorder(tab, index)
  local new_tab = {}
  for i = 1, #tab do
    table.insert(new_tab, tab[index[i]])
  end
  return new_tab
end

return {
  reorder = reorder,
  zero = zero,
  append = append
}
