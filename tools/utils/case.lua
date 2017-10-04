
local unicode = require('tools.utils.unicode')
local separators = require('tools.utils.separators')

local case = {}

function case.combineCase(feat, thecase)
  if feat == 'N' then
    if thecase == 'lower' then feat = 'L' end
    if thecase == 'upper' then feat = 'C1' end
  elseif feat == 'L' then
    if thecase == 'upper' then feat = 'M' end
  elseif feat == 'C1' then
    if thecase == 'upper' then feat = 'U' end
    if thecase == 'lower' then feat = 'C' end
  elseif feat == 'C' then
    if thecase == 'upper' then feat = 'M' end
  elseif feat == 'U' then
    if thecase == 'lower' then feat = 'M' end
  end
  return feat
end

-- add case feature to tokens
function case.addCase (toks)
  for i=1, #toks do
    local casefeat = 'N'
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
    local casefeat = 'N'
    local newTok = ''

    for v, c in unicode.utf8_iter(toks[i]) do
      local is_letter, thecase = unicode.isLetter(v)
      if is_letter then
        if case.combineCase(casefeat, thecase) == 'M' then
          table.insert(caseSegment, newTok..separator)
          newTok = ''
          casefeat = case.combineCase('N', thecase)
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
  if feats[1] == 'L' or feats[1] == 'N' then
    return w
  else
    local wr = ''
    for v, c in unicode.utf8_iter(w) do
      if wr == '' or feats[1] == 'U' then
        local _, cu = unicode.getUpper(v)
        if cu then c = cu end
      end
      wr = wr .. c
    end
    return wr
  end
end

function case.getFeatures()
  return {'C', 'l', 'n', 'U', 'M'}
end

return case
