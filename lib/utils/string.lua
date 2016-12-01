--[[
  Split `str` on string or pattern separator `sep`.
  Compared to the standard Lua split function, this one does not drop empty fragment.
]]
local function split(str, sep)
  local res = {}
  local index = 1

  while index <= str:len() do
    local sep_start, sep_end = str:find(sep, index)

    local sub
    if not sep_start then
      sub = str:sub(index)
      table.insert(res, sub)
      index = str:len() + 1
    else
      sub = str:sub(index, sep_start - 1)
      table.insert(res, sub)
      index = sep_end + 1
      if index > str:len() then
        table.insert(res, '')
      end
    end
  end

  return res
end

--[[ Remove whitespaces at the start and end of the string `s`. ]]
local function strip(s)
  return s:gsub("^%s+",""):gsub("%s+$","")
end

--[[ Convenience function to test `s` for emptiness. ]]
local function is_empty(s)
  return s == nil or s == ''
end

return {
  split = split,
  strip = strip,
  is_empty = is_empty
}
