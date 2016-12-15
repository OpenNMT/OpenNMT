unicode = require './utils/unicode'

local cmd = torch.CmdLine()

cmd:text("")
cmd:text("**tokenize.lua**")
cmd:text("")

cmd:option('-mode', 'aggressive', [[Define how aggressive should the tokenization be - 'aggressive' is only keep sequences of letters/numbers,
                                    'conservative' allows mix of alphanumeric]])
cmd:option('-sep_feature', false, [[Generate separator feature]])
cmd:option('-case_feature', false, [[Generate case feature]])

local opt = cmd:parse(arg)

local function combineCase(feat, case)
  if feat == 'N' then
    if case == 'lower' then feat = 'L' end
    if case == 'upper' then feat = 'C1' end
  elseif feat == 'L' then
    if case == 'upper' then feat = 'M' end
  elseif feat == 'C1' then
    if case == 'upper' then feat = 'U' end
    if case == 'lower' then feat = 'C' end
  elseif feat == 'C' then
    if case == 'upper' then feat = 'M' end
  end
  return feat
end

-- minimalistic tokenization
-- - remove utf-8 BOM character
-- - turn sequences of separators into single space
-- - skip any other non control character [U+0001-U+002F]
-- - keep sequence of letters/numbers and tokenize everything else

function tokenize(line)
  local nline = ""
  local spacefeat = 'N'
  local casefeat = 'N'
  local space = true
  local letter = false
  local number = false
  for v, c, nextv in unicode.utf8_iter(line) do
    if unicode.isSeparator(v) then
      if space == false then
        if opt.sep_feature then nline = nline..'-|-'..spacefeat end
        if opt.case_feature then nline = nline..'-|-'..string.sub(casefeat,1,1) end
        nline = nline..' '
      end
      number = false
      letter = false
      space = true
      spacefeat = 'S'
      casefeat = 'N'
      last = ' '
    else
      if v > 32 and not(v == 0xFEFF) then
        local is_letter, case = unicode.isLetter(v)
        local is_number = unicode.isNumber(v)
        if opt.mode == 'conservative' then
          if is_number or (c == '-' and letter == true) or c == '_' or
             (letter == true and (c == '.' or c == ',') and (unicode.isNumber(nextv) or unicode.isLetter(nextv))) then
            is_letter = true
            case = "other"
          end
        end
        if is_letter then
          if not(letter == true or space == true) then
            if opt.sep_feature then nline = nline..'-|-'..spacefeat end
            if opt.case_feature then nline = nline..'-|-'..string.sub(casefeat,1,1) end
            nline = nline..' '
            spacefeat = 'N'
            casefeat = 'N'
          end
          casefeat = combineCase(casefeat, case)
          nline = nline..c
          space = false
          number = false
          letter = true
        elseif is_number then
          if not(number == true or space == true) then
            if opt.sep_feature then nline = nline..'-|-'..spacefeat end
            if opt.case_feature then nline = nline..'-|-'..string.sub(casefeat,1,1) end
            nline = nline..' '
            spacefeat = 'N'
            casefeat = 'N'
          end
          nline = nline..c
          space = false
          letter = false
          number = true
        else
          if not space == true then
            if opt.sep_feature then nline = nline..'-|-'..spacefeat end
            if opt.case_feature then nline = nline..'-|-'..string.sub(casefeat,1,1) end
            nline = nline .. ' '
            spacefeat = 'N'
            casefeat = 'N'
          end
          nline = nline..c
          if opt.sep_feature then nline = nline..'-|-'..spacefeat end
          if opt.case_feature then nline = nline..'-|-'..string.sub(casefeat,1,1) end
          nline = nline..' '
          number = false
          letter = false
          space = true
        end
      end
    end
  end

  -- remove final space
  if space == true then
    nline = string.sub(nline, 1, -2)
  else
    if opt.sep_feature then nline = nline..'-|-'..spacefeat end
    if opt.case_feature then nline = nline..'-|-'..string.sub(casefeat,1,1) end
  end

  return nline
end

local timer = torch.Timer()
local idx = 1
for line in io.lines() do
  local res, err = pcall(function() io.write(tokenize(line) .. '\n') end)
  if not res then
    if string.find(err,"interrupted") then
      error("interrupted")
    else
      error("unicode error in line "..idx..": "..line..'-'..err)
    end
  end
  idx = idx + 1
end

io.stderr:write(string.format('Tokenization completed in %0.3f seconds - %d sentences\n',timer:time().real,idx-1))
