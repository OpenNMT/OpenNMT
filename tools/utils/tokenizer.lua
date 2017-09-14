local tokenizer = {}

local unicode = require('tools.utils.unicode')
local case = require ('tools.utils.case')
local separators = require('tools.utils.separators')
local alphabet = require('tools.utils.alphabets')

local alphabet_list = {}
for k,_ in pairs(alphabet.ranges) do
  table.insert(alphabet_list, k)
end

local options = {
  {
    '-mode', 'conservative',
    [[Define how aggressive should the tokenization be. `aggressive` only keeps sequences
      of letters/numbers, `conservative` allows a mix of alphanumeric as in: "2,000", "E65",
      "soft-landing", etc. `space` is doing space tokenization.]],
    {
      enum = {'space', 'conservative', 'aggressive'}
    }
  },
  {
    '-joiner_annotate', false,
    [[Include joiner annotation using `-joiner` character.]]
  },
  {
    '-joiner', separators.joiner_marker,
    [[Character used to annotate joiners.]]
  },
  {
    '-joiner_new', false,
    [[In `-joiner_annotate` mode, `-joiner` is an independent token.]]
  },
  {
    '-case_feature', false,
    [[Generate case feature.]]
  },
  {
    '-segment_case', false,
    [[Segment case feature, splits AbC to Ab C to be able to restore case]]
  },
  {
    '-segment_alphabet', {},
    [[Segment all letters from indicated alphabet.]],
    {
      enum = alphabet_list,
    }
  },
  {
    '-segment_alphabet_change', false,
    [[Segment if alphabet change between 2 letters.]]
  },
  {
    '-bpe_model', '',
    [[Apply Byte Pair Encoding if the BPE model path is given. If the option is used,
      BPE related options will be overridden/set automatically if the BPE model specified by `-bpe_model`
      is learnt using `learn_bpe.lua`.]]
  },
  {
    '-EOT_marker', separators.EOT,
    [[Marker used to mark the end of token.]]
  },
  {
    '-BOT_marker', separators.BOT,
    [[Marker used to mark the beginning of token.]]
  },
  {
    '-bpe_case_insensitive', false,
    [[Apply BPE internally in lowercase, but still output the truecase units.
      This option will be overridden/set automatically if the BPE model specified by `-bpe_model`
      is learnt using `learn_bpe.lua`.]]
  },
  {
    '-bpe_mode', 'suffix',
    [[Define the BPE mode. This option will be overridden/set automatically if the BPE model
      specified by `-bpe_model` is learnt using `learn_bpe.lua`. `prefix`: append `-BOT_marker`
      to the begining of each word to learn prefix-oriented pair statistics;
      `suffix`: append `-EOT_marker` to the end of each word to learn suffix-oriented pair
      statistics, as in the original Python script; `both`: `suffix` and `prefix`; `none`:
      no `suffix` nor `prefix`.]],
    {
      enum = {'suffix', 'prefix', 'both', 'none'}
    }
  }
}

function tokenizer.getOpts()
  return options
end

function tokenizer.declareOpts(cmd)
  cmd:setCmdLineOptions(options, 'Tokenizer')
end

local function inTable(v, t)
  for _, vt in ipairs(t) do
    if v == vt then
      return true
    end
  end
  return false
end

-- minimalistic tokenization
-- - remove utf-8 BOM character
-- - turn sequences of separators into single space
-- - skip any other non control character [U+0001-U+002F]
-- - keep sequence of letters/numbers and tokenize everything else
local function tokenize(line, opt)

  if opt.mode == 'space' then
    local index = 1
    local tokens = {}
    while index <= line:len() do
      local sepStart, sepEnd = line:find(' ', index)
      local sub
      if not sepStart then
        sub = line:sub(index)
        table.insert(tokens, sub)
        break
      else
        sub = line:sub(index, sepStart - 1)
        table.insert(tokens, sub)
        index = sepEnd + 1
      end
    end

    return tokens
  end

  local tokens = {}
  -- contains the current token
  local curtok = ''
  -- keep category of the previous character
  local space = true
  local letter = false
  local prev_alphabet
  local number = false
  local other = false
  local placeholder = false

  -- iterate on utf-8 characters
  for v, c, nextv in unicode.utf8_iter(line) do
    if placeholder then
      if c == separators.ph_marker_close then
        curtok = curtok .. c
        letter = true
        prev_alphabet = 'placeholder'
        placeholder = false
        space = false
      else
        if unicode.isSeparator(v) then
          c = string.format(separators.protected_character.."%04x", v)
        end
        curtok = curtok .. c
      end
    elseif c == separators.ph_marker_open then
      if space == false then
        if opt.joiner_annotate and not(opt.joiner_new) then
          curtok = curtok .. opt.joiner
        end
        table.insert(tokens, curtok)
        if opt.joiner_annotate and opt.joiner_new then
          table.insert(tokens, opt.joiner)
        end
      elseif other == true then
        if opt.joiner_annotate then
          if curtok == '' then
            if opt.joiner_new then table.insert(tokens, opt.joiner)
            else tokens[#tokens] = tokens[#tokens] .. opt.joiner end
          end
        end
      end
      curtok = c
      placeholder = true
    elseif unicode.isSeparator(v) then
      if space == false then
        table.insert(tokens, curtok)
        curtok = ''
      end
      -- if the character is the ZERO-WIDTH JOINER character (ZWJ), add joiner
      if v == 0x200D then
        if opt.joiner_annotate and opt.joiner_new and #tokens then
          table.insert(tokens, opt.joiner)
        elseif opt.joiner_annotate then
          if other or (number and unicode.isLetter(nextv)) then
            tokens[#tokens] = tokens[#tokens] .. opt.joiner
          else
            curtok = opt.joiner
          end
        end
      end
      number = false
      letter = false
      space = true
      other = false
    else
      -- skip special characters and BOM and
      if v > 32 and not(v == 0xFEFF) then
        -- normalize the separator marker and feat separator
        if c == separators.joiner_marker then c = separators.joiner_marker_substitute end
        if c == separators.feat_marker then c = separators.feat_marker_substitute end


        local is_letter = unicode.isLetter(v)
        local is_alphabet
        if is_letter and (opt.segment_alphabet_change or #opt.segment_alphabet>0) then
          is_alphabet = alphabet.findAlphabet(v)
        end

        local is_number = unicode.isNumber(v)
        local is_mark = unicode.isMark(v)
        -- if we have a mark, we keep type of previous character
        if is_mark then
          is_letter = letter
          is_number = number
        end
        if opt.mode == 'conservative' then
          if is_number or (c == '-' and letter == true) or c == '_' or
                (letter == true and (c == '.' or c == ',') and (unicode.isNumber(nextv) or unicode.isLetter(nextv))) then
            is_letter = true
          end
        end
        if is_letter then
          if not(letter == true or space == true) or
             (letter == true and not is_mark and
              (prev_alphabet == 'placeholder' or
               (prev_alphabet == is_alphabet and inTable(is_alphabet, opt.segment_alphabet)) or
               (prev_alphabet ~= is_alphabet and opt.segment_alphabet_change))) then
            if opt.joiner_annotate and not(opt.joiner_new) and prev_alphabet ~= 'placeholder' then
              curtok = curtok .. opt.joiner
            end
            table.insert(tokens, curtok)
            if opt.joiner_annotate and opt.joiner_new then
              table.insert(tokens, opt.joiner)
            end
            curtok = ''
            if opt.joiner_annotate and not(opt.joiner_new) and prev_alphabet == 'placeholder' then
              curtok = curtok .. opt.joiner
            end
          elseif other == true then
            if opt.joiner_annotate then
              if curtok == '' then
                if opt.joiner_new then table.insert(tokens, opt.joiner)
                else tokens[#tokens] = tokens[#tokens] .. opt.joiner end
             end
           end
          end
          curtok = curtok .. c
          space = false
          number = false
          other = false
          letter = true
          prev_alphabet = is_alphabet
        elseif is_number then
          if letter == true or not(number == true or space == true) then
            local addjoiner = false
            if opt.joiner_annotate then
              if opt.joiner_new then
                addjoiner = true
              else
                if not(letter) and not(placeholder) then
                  curtok = curtok .. opt.joiner
                else
                  c = opt.joiner .. c
                end
              end
            end
            table.insert(tokens, curtok)
            if addjoiner then
              table.insert(tokens, opt.joiner)
            end
            curtok = ''
          elseif other == true then
            if opt.joiner_annotate then
              if opt.joiner_new then
                table.insert(tokens, opt.joiner)
              else
                tokens[#tokens] = tokens[#tokens] .. opt.joiner
              end
            end
          end
          curtok = curtok..c
          space = false
          letter = false
          other = false
          number = true
        else
          if not space == true then
            if opt.joiner_annotate and not(opt.joiner_new) then
              c = opt.joiner .. c
            end
            table.insert(tokens, curtok)
            if opt.joiner_annotate and opt.joiner_new then
              table.insert(tokens, opt.joiner)
            end
            curtok = ''
          elseif other == true then
            if opt.joiner_annotate then
              if opt.joiner_new then
                table.insert(tokens, opt.joiner)
              else
                curtok = opt.joiner
              end
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

function tokenizer.tokenize(opt, line, bpe)
  -- tokenize
  local tokens = tokenize(line, opt)

  -- apply segmetn feature if requested
  if opt.segment_case then
    local sep = ''
    if opt.joiner_annotate then sep = opt.joiner end
    tokens = case.segmentCase(tokens, sep)
  end

  -- apply bpe if requested
  if bpe then
    local sep = ''
    if opt.joiner_annotate then sep = opt.joiner end
    tokens = bpe:segment(tokens, sep)
  end

  -- add-up case feature if requested
  if opt.case_feature then
    tokens = case.addCase(tokens)
  end

  return tokens
end

local function analyseToken(t, joiner)
  local feats = {}
  local tok = ""
  local p
  local leftsep = false
  local rightsep = false
  local i = 1
  while i <= #t do
    if t:sub(i, i+#separators.feat_marker-1) == separators.feat_marker then
      p = i
      break
    end
    tok = tok .. t:sub(i, i)
    i = i + 1
  end
  if tok:sub(1,#joiner) == joiner then
    tok = tok:sub(1+#joiner)
    leftsep = true
    if tok == '' then rightsep = true end
  end
  if tok:sub(-#joiner,-1) == joiner then
    tok = tok:sub(1,-#joiner-1)
    rightsep = true
  end
  if p then
    p = p + #separators.feat_marker
    local j = p
    while j <= #t do
      if t:sub(j, j+#separators.feat_marker-1) == separators.feat_marker then
        table.insert(feats, t:sub(p, j-1))
        j = j + #separators.feat_marker - 1
        p = j + 1
      end
      j = j + 1
    end
    table.insert(feats, t:sub(p))
  end
  return tok, leftsep, rightsep, feats
end

local function getTokens(t, joiner)
  local fields = {}
  t:gsub("([^ ]+)", function(tok)
    local w, leftsep, rightsep, feats =  analyseToken(tok, joiner)
    table.insert(fields, { w=w, leftsep=leftsep, rightsep=rightsep, feats=feats })
  end)
  return fields
end

function tokenizer.detokenize(line, opt)
  local dline = ""
  local tokens = getTokens(line, opt.joiner)
  for j = 1, #tokens do
    local token = tokens[j].w
    if j > 1 and not tokens[j-1].rightsep and not tokens[j].leftsep then
      dline = dline .. " "
    end
    if token:sub(1, separators.ph_marker_open:len()) == separators.ph_marker_open then
      local inProtected = false
      local protectSeq = ''
      local rtok = ''
      for _, c, _ in unicode.utf8_iter(token) do
        if inProtected then
          protectSeq = protectSeq .. c
          if protectSeq:len() == 4 then
            rtok = rtok .. unicode._cp_to_utf8(tonumber(protectSeq, 16))
            inProtected = false
          end
        elseif c == separators.protected_character then
          inProtected = true
        else
          rtok = rtok .. c
          if c == separators.ph_marker_close then
            break
          end
        end
      end
      token = rtok
    end
    if opt.case_feature then
      token = case.restoreCase(token, tokens[j].feats)
    end
    dline = dline .. token
  end
  return dline
end

return tokenizer
