
local unicode = require('tools.utils.unicode')
local separators = require('tools.utils.separators')

local case = {}

case.CAPITAL = 'C'
case.CAPITAL_FIRST = 'C1'
case.LOWER = 'L'
case.MIXED = 'M'
case.NONE = 'N'
case.UPPER = 'U'

function case.combineCase(feat, thecase)
  if feat == case.NONE then
    if thecase == 'lower' then feat = case.LOWER end
    if thecase == 'upper' then feat = case.CAPITAL_FIRST end
  elseif feat == case.LOWER then
    if thecase == 'upper' then feat = case.MIXED end
  elseif feat == case.CAPITAL_FIRST then
    if thecase == 'upper' then feat = case.UPPER end
    if thecase == 'lower' then feat = case.CAPITAL end
  elseif feat == case.CAPITAL then
    if thecase == 'upper' then feat = case.MIXED end
  elseif feat == case.UPPER then
    if thecase == 'lower' then feat = case.MIXED end
  end
  return feat
end

-- add case feature to tokens
function case.addCase (toks)
  for i=1, #toks do
    local casefeat = case.NONE
    local loweredTok = ''

    for v, c in unicode.utf8_iter(toks[i]) do
      if loweredTok == '' and c == separators.ph_marker_open then
        loweredTok = toks[i]
        break
      end
      local is_letter, thecase = unicode.isLetter(v)
      -- find lowercase equivalent
      if is_letter then
        local lu, lc = unicode.getLower(v)
        if lu then c = lc end
        casefeat = case.combineCase(casefeat, thecase)
      end
      loweredTok = loweredTok..c
    end

    toks[i] = loweredTok..separators.feat_marker..string.sub(casefeat,1,1)
  end
  return toks
end

-- Segment tokens by case in order to have only Abc abc ABC words
function case.segmentCase (toks, separator)
  local caseSegment = {}
  for i=1, #toks do
    local casefeat = case.NONE
    local newTok = ''

    for v, c in unicode.utf8_iter(toks[i]) do
      local is_letter, thecase = unicode.isLetter(v)
      if is_letter then
        if case.combineCase(casefeat, thecase) == case.MIXED then
          table.insert(caseSegment, newTok..separator)
          newTok = ''
          casefeat = case.combineCase(case.NONE, thecase)
        else
          casefeat = case.combineCase(casefeat, thecase)
        end
      end
      newTok = newTok..c
    end
    table.insert(caseSegment, newTok)
  end
  return caseSegment
end

function case.restoreCase(w, feats)
  assert(#feats>=1)
  if feats[1] == case.LOWER or feats[1] == case.NONE then
    return w
  else
    local wr = ''
    for v, c in unicode.utf8_iter(w) do
      if wr == '' or feats[1] == case.UPPER then
        local _, cu = unicode.getUpper(v)
        if cu then c = cu end
      end
      wr = wr .. c
    end
    return wr
  end
end

function case.getFeatures()
  return {
    case.CAPITAL,
    case.LOWER,
    case.NONE,
    case.UPPER,
    case.MIXED
  }
end

return case
