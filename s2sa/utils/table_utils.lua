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

local function copy(orig)
  local orig_type = type(orig)
  local copy_obj
  if orig_type == 'table' then
    copy_obj = {}
    for orig_key, orig_value in pairs(orig) do
      copy_obj[orig_key] = orig_value
    end
  else
    copy_obj = orig
  end
  return copy_obj
end

-- Convert a flat index to a row-column tuple.
local function flat_to_rc(v, flat_index)
  local row = math.floor((flat_index - 1) / v:size(2)) + 1
  return row, (flat_index - 1) % v:size(2) + 1
end

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
  append = append,
  copy = copy,
  flat_to_rc = flat_to_rc
}
