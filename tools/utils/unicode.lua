-- for lua < 5.3 compatibility
if not bit32 then
  bit32 = require 'bit32'
end

unidata = require './unidata'

unicode = {}

-- convert the next utf8 character to ucs
-- returns codepoint and utf-8 character
function _utf8_to_cp(s, idx)
  idx = idx or 1
  local c = string.byte(s, idx)
  local l = (c < 0x80 and 1) or (c < 0xE0 and 2) or (c < 0xF0 and 3) or (c < 0xF8 and 4)
  if not l then error("invalid utf-8 sequence") end
  local val = 0
  if l == 1 then return c, string.sub(s, idx, idx) end
  for i = 1, l do
    c = string.byte(s, idx+i-1)
    if i > 1 then
      assert(bit32.band(c, 0xC0) == 0x80)
      val = bit32.lshift(val, 6)
      val = bit32.bor(val, bit32.band(c, 0x3F))
    else
      val = bit32.band(c,bit32.rshift(0xff,l))
    end
  end
  return val, string.sub(s, idx, idx+l-1)
end

-- convert unicode codepoint to utf8
function _cp_to_utf8(u)
  assert(u>=0 and u<=0x10FFFF)
  if u <= 0x7F then
    return string.char(u)
  elseif u <= 0x7FF then
    local b0 = 0xC0 + bit32.rshift(u, 6)
    local b1 = 0x80 + bit.band(u, 0x3F)
    return string.char(b0, b1)
  elseif u <= 0xFFFF then
    local b0 = 0xE0 + bit32.rshift(u, 12)
    local b1 = 0x80 + bit32.band(bit32.rshift(u, 6), 0x3f)
    local b2 = 0x80 + bit32.band(u, 0x3f)
    return string.char(b0, b1, b2)
  end
  local b0 = 0xF0 + bit32.rshift(u, 18)
  local b1 = 0x80 + bit32.band(bit32.rshift(u, 12), 0x3f)
  local b2 = 0x80 + bit32.band(bit32.rshift(u, 6), 0x3f)
  local b3 = 0x80 + bit32.band(u, 0x3f)
  return string.char(b0, b1, b2, b3)
end

function unicode.utf8_iter(s)
  local p = 1
  local L = #s
  return function()
    if p > L then return end
    local v, c = _utf8_to_cp(s, p)
    p = p + #c
    return v, c
  end
end

local function _find_codepoint(u, utable)
  for i,v in pairs(utable) do
    if u >= i then
      local idx = bit32.rshift(u-i,4) + 1
      local p = (u-i) % 16
      if v[idx] then
        return not(bit.band(bit32.lshift(v[idx], p), 0x8000) == 0)
      end
    end
  end
  return false
end

function unicode.isSeparator(u)
  -- control character or separator
  return (u >= 9 and u <= 13) or _find_codepoint(u, unidata.Separator)
end

-- returns if letter and case "lower", "upper", "other"
function unicode.isLetter(u)
  -- unicode letter or CJK Unified Ideograph
  if ((u>=0x4E00 and u<=0x9FD5) -- CJK Unified Ideograph
      or (u>=0x2F00 and u<=0x2FD5) -- Kangxi Radicals
      or (u>=0x2E80 and u<=0x2EFF) -- CJK Radicals Supplement
      or (u>=0x3040 and u<=0x319F) -- Hiragana, Katakana, Bopomofo, Hangul, Kanbun
      or _find_codepoint(u, unidata.LetterOther)
      ) then
    return true, "other"
  end
  if _find_codepoint(u, unidata.LetterLower) then
    return true, "lower"
  end
  if _find_codepoint(u, unidata.LetterUpper) then
    return true, "upper"
  end
  return false
end

function unicode.isNumber(u)
  return _find_codepoint(u, unidata.Number)
end

function unicode.isAlnum(u)
  return unicode.isLetter(u) or unicode.isNumber(u) or u=='_'
end

return unicode
