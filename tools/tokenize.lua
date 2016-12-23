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
cmd:option('-sep_annotate', false, [[Include separator annotation using sep_marker]])
cmd:option('-case_feature', false, [[Generate case feature]])
cmd:option('-bpe_model', '', [[Apply Byte Pair Encoding if the BPE model path is given]])

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
  -- contains the current token
  local curtok = ''
  -- keep category of the previous character
  local space = true
  local letter = false
  local number = false
  local other = false

  -- iterate on utf-8 characters
  for v, c, nextv in unicode.utf8_iter(line) do
    if unicode.isSeparator(v) then
      if space == false then
        table.insert(tokens, curtok)
        curtok = ''
      end
      number = false
      letter = false
      space = true
      other = false
    else
      -- skip special charactes and BOM and
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
          if opt.sep_annotate then
            curtok = appendMarker(curtok)
          end
          table.insert(tokens, curtok)
          curtok = ''
          elseif other == true then
            if opt.sep_annotate then
              if (curtok == '') then
               tokens[#tokens] = tokens[#tokens] .. sep_marker
             end
           end
          end
          curtok = curtok .. c
          space = false
          number = false
          other = false
          letter = true
        elseif is_number then
          if not(number == true or space == true) then
            if opt.sep_annotate then
              if not(letter) then
                curtok = appendMarker(curtok)
              else
                c = sep_marker .. c
              end
            end
            table.insert(tokens, curtok)
            curtok = ''
          elseif other == true then
            if opt.sep_annotate then
              curtok = appendMarker(curtok)
            end
          end
          curtok = curtok..c
          space = false
          letter = false
          other = false
          number = true
        else
          if not space == true then
            if opt.sep_annotate then
              c = sep_marker .. c
            end
            table.insert(tokens, curtok)
            curtok = ''
          elseif other == true then
            if opt.sep_annotate then
              c = sep_marker .. c
            end
          end
          curtok = curtok .. c
          table.insert(tokens, curtok)
          curtok = ''
          number = false
          letter = false
          other = true
          space = true
        end
      end
    end
  end

  -- last token
  if (curtok ~= '') then
    table.insert(tokens, curtok)
  end

  return tokens
end

-- add case feature to tokens
local function addCase (toks)
  for i=1, #toks do
    local casefeat = 'N'
    local loweredTok = ''

    for v, c in unicode.utf8_iter(toks[i]) do
      local is_letter, case = unicode.isLetter(v)
      -- find lowercase equivalent
      if is_letter then
        local lu, lc = unicode.getLower(v)
        if lu then c = lc end
        casefeat = combineCase(casefeat, case)
      end
      loweredTok = loweredTok..c
    end
    toks[i] = loweredTok..feat_marker..string.sub(casefeat,1,1)
  end
  return toks
end

local timer = torch.Timer()
local idx = 1
local bpe

if opt.bpe_model ~= '' then
  bpe = BPE.new(opt.bpe_model)
end

for line in io.lines() do
  local res
  local err

  -- tokenize
  local tokens
  res, err = pcall(function() tokens = tokenize(line) end)

  -- it can generate an exception if there are utf-8 issues in the text
  if not res then
    if string.find(err, "interrupted") then
      error("interrupted")
    else
      error("unicode error in line " .. idx .. ": " .. line .. '-' .. err)
    end
  end

  -- apply bpe if requested
  if bpe then
    tokens = bpe:segment(tokens, sep_marker)
  end

  -- add-up case feature if requested
  if opt.case_feature then
    tokens = addCase(tokens)
  end

  -- output the tokenized and featurized string
  io.write(table.concat(tokens, ' ') .. '\n')

  idx = idx + 1
end

io.stderr:write(string.format('Tokenization completed in %0.3f seconds - %d sentences\n', timer:time().real, idx-1))
