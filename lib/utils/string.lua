local function strip(s)
  return s:gsub("^%s+",""):gsub("%s+$","")
end

local function is_empty(s)
  return s == nil or s == ''
end

return {
  strip = strip,
  is_empty = is_empty
}
