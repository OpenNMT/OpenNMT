require('torch')

local unicode = require('tools.utils.unicode')
local BPE = require ('tools.utils.BPE')

local cmd = torch.CmdLine()

local sep_marker = '\\@'
local feat_marker = '\\|'
local protect_char = '\\'

cmd:text("")
cmd:text("**tokenize.lua**")
cmd:text("")

cmd:option('-mode', 'conservative', [[Define how aggressive should the tokenization be - 'aggressive' only keeps sequences of letters/numbers,
                                    'conservative' allows mix of alphanumeric as in: '2,000', 'E65', 'soft-landing']])
cmd:option('-sep_annotate', 'marker', [[Include separator annotation using sep_marker (marker), or feature (feature), or nothing (none)]])
cmd:option('-case_feature', false, [[Generate case feature]])
cmd:option('-bpe', '', [[Apply BPE if the BPE model path is given]])

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

local function appendMarker(l)
  if opt.case_feature then
    local p=l:find(feat_marker, -4)
    return l:sub(1,p-1)..sep_marker..l:sub(p)
  end
  return l..sep_marker
end

-- minimalistic tokenization
-- - remove utf-8 BOM character
-- - turn sequences of separators into single space
-- - skip any other non control character [U+0001-U+002F]
-- - keep sequence of letters/numbers and tokenize everything else

local function tokenize(line)
  local tokens = {}
  local nline = ""
  local space = true
  local letter = false
  local number = false
  local other = false
  for v, c, nextv in unicode.utf8_iter(line) do
    if unicode.isSeparator(v) then
      if space == false then
        table.insert(tokens, nline)
	nline = ""
      end
      number = false
      letter = false
      space = true
      other = false
    else
      if v > 32 and not(v == 0xFEFF) then
        if c == protect_char then c = protect_char..c end
        local is_letter, _ = unicode.isLetter(v)
        local is_number = unicode.isNumber(v)
        if opt.mode == 'conservative' then
          if is_number or (c == '-' and letter == true) or c == '_' or
             (letter == true and (c == '.' or c == ',') and (unicode.isNumber(nextv) or unicode.isLetter(nextv))) then
            is_letter = true
          end
        end
        if is_letter then
          if not(letter == true or space == true) then
            if opt.sep_annotate == 'marker' then
              nline = appendMarker(nline)
            end
            table.insert(tokens, nline)
	    nline = ''
          elseif other == true then
            if opt.sep_annotate == 'marker' then
	      if (nline == '') then
	        tokens[#tokens]=tokens[#tokens]..sep_marker
              end
            end
          end
          nline = nline..c
          space = false
          number = false
          other = false
          letter = true
        elseif is_number then
          if not(number == true or space == true) then
            if opt.sep_annotate == 'marker' then
              if not(letter) then
                nline = appendMarker(nline)
              else
                c = sep_marker..c
              end
            end
            table.insert(tokens, nline)
	    nline = ''
          elseif other == true then
            if opt.sep_annotate == 'marker' then
              nline = appendMarker(nline)
            end
          end
          nline = nline..c
          space = false
          letter = false
          other = false
          number = true
        else
          if not space == true then
            if opt.sep_annotate == 'marker' then
              c = sep_marker..c
            end
            table.insert(tokens, nline)
	    nline = ''
          elseif other == true then
            if opt.sep_annotate == 'marker' then
              c = sep_marker..c
            end
          end
          nline = nline..c
	  table.insert(tokens, nline)
	  nline = ''
          number = false
          letter = false
          other = true
          space = true
        end
      end
    end
  end
  if (nline ~= '') then
    table.insert(tokens, nline)
  end
  return tokens
end

local function addCase (toks)
  for i=1, #toks do
    local casefeat = 'N'
    local letter = false
    local number = false
    local other = false
    local loweredTok = ''

    for v, c, nextv in unicode.utf8_iter(toks[i]) do
      if v > 32 and not(v == 0xFEFF) then
        local is_letter, case = unicode.isLetter(v)
        if is_letter then
          local lu, lc = unicode.getLower(v)
          if lu then c = lc end
        end
        local is_number = unicode.isNumber(v)
        if opt.mode == 'conservative' then
          if is_number or (c == '-' and letter == true) or c == '_' or
             (letter == true and (c == '.' or c == ',') and (unicode.isNumber(nextv) or unicode.isLetter(nextv))) then
            is_letter = true
            case = "other"
          end
        end
        if is_letter then
          if not(letter == true) then
	    casefeat = 'N'
          end
          casefeat = combineCase(casefeat, case)
	  loweredTok = loweredTok..c
          number = false
          other = false
          letter = true
        elseif is_number then
	  loweredTok = loweredTok..c
          letter = false
          other = false
          number = true
        else
	  loweredTok = loweredTok..c
          number = false
          letter = false
          other = true
        end
      end
    end
    toks[i] = loweredTok..feat_marker..string.sub(casefeat,1,1)
  end
  return toks

end

local timer = torch.Timer()
local idx = 1
local bpe = nil

if (opt.bpe ~= '') then
  bpe = BPE.new(opt.bpe)
end


for line in io.lines() do
  local res = true
  local err = ''
  if (bpe ~= nil) then
    if (opt.case_feature) then
      res, err = pcall(function() io.write(table.concat(addCase(bpe:segment(tokenize(line))), ' ') .. '\n') end)
    else
      res, err = pcall(function() io.write(table.concat(bpe:segment(tokenize(line)), ' ') .. '\n') end)
    end
  elseif (opt.case_feature) then
    res, err = pcall(function() io.write(table.concat(addCase(tokenize(line)), ' ') .. '\n') end)
  else
    res, err = pcall(function() io.write(table.concat(tokenize(line), ' ') .. '\n') end)
  end

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
