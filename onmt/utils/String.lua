--[[
  Split `str` on string or pattern separator `sep`.
  Compared to the standard Lua split function, this one does not drop empty fragment.
]]
local function split(str, sep)
  local res = {}
  local index = 1

  while index <= str:len() do
    local sepStart, sepEnd = str:find(sep, index)

    local sub
    if not sepStart then
      sub = str:sub(index)
      table.insert(res, sub)
      index = str:len() + 1
    else
      sub = str:sub(index, sepStart - 1)
      table.insert(res, sub)
      index = sepEnd + 1
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

--[[ Remove initial hyphen(s). ]]
local function stripHyphens(str)
   return string.match(str, '%-*(.*)')
end

--[[ Right pad a strip with spaces. ]]
local function pad(str, sz)
   return str .. string.rep(' ', sz-#str)
end

--[[ Convenience function to test `s` for emptiness. ]]
local function isEmpty(s)
  return s == nil or s == ''
end

return {
  split = split,
  strip = strip,
  isEmpty = isEmpty,
  pad = pad,
  stripHyphens = stripHyphen
}
