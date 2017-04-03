local tokenizer = {}

local unicode = require('tools.utils.unicode')
local case = require ('tools.utils.case')
local separators = require('tools.utils.separators')

local options = {
  {
    '-mode', 'conservative',
    [[Define how aggressive should the tokenization be - 'aggressive' only keeps sequences of letters/numbers,
    'conservative' allows mix of alphanumeric as in: '2,000', 'E65', 'soft-landing']]
  },
  {
    '-joiner_annotate', false,
    [[Include joiner annotation using 'joiner' character]]
  },
  {
    '-joiner', separators.joiner_marker,
    [[Character used to annotate joiners]]
  },
  {
    '-joiner_new', false,
    [[in joiner_annotate mode, 'joiner' is an independent token]]
  },
  {
    '-case_feature', false,
    [[Generate case feature]]
  },
  {
    '-bpe_model', '',
    [[Apply Byte Pair Encoding if the BPE model path is given. If the option is used, 'mode' will be overridden/set automatically if the BPE model specified by bpe_model is learnt using learn_bpe.lua]]
  },
  {
    '-EOT_marker', separators.EOT,
    [[Marker used to mark the end of token, use '</w>' for python models, otherwise default value ]]
  },
  {
    '-BOT_marker', separators.BOT,
    [[Marker used to mark the begining of token]]
  },
  {
    '-bpe_case_insensitive', false,
    [[Apply BPE internally in lowercase, but still output the truecase units. This option will be overridden/set automatically if the BPE model specified by bpe_model is learnt using learn_bpe.lua]]
  },
  {
    '-bpe_mode', 'suffix',
    [[Define the mode for bpe. This option will be overridden/set automatically if the BPE model specified by bpe_model is learnt using learn_bpe.lua. - 'prefix': Append '﹤' to the begining of each word to learn prefix-oriented pair statistics;
    'suffix': Append '﹥' to the end of each word to learn suffix-oriented pair statistics, as in the original python script;}
    'both': suffix and prefix; 'none': no suffix nor prefix]]
  }
}

function tokenizer.declareOpts(cmd)
  cmd:setCmdLineOptions(options, 'Tokenizer')
end

-- minimalistic tokenization
-- - remove utf-8 BOM character
-- - turn sequences of separators into single space
-- - skip any other non control character [U+0001-U+002F]
-- - keep sequence of letters/numbers and tokenize everything else
local function tokenize(line, opt)
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
        local is_number = unicode.isNumber(v)
        if opt.mode == 'conservative' then
          if is_number or (c == '-' and letter == true) or c == '_' or
                (letter == true and (c == '.' or c == ',') and (unicode.isNumber(nextv) or unicode.isLetter(nextv))) then
            is_letter = true
          end
      end
      if is_letter then
        if not(letter == true or space == true) then
          if opt.joiner_annotate and not(opt.joiner_new) then
            curtok = curtok .. opt.joiner
          end
          table.insert(tokens, curtok)
          if opt.joiner_annotate and opt.joiner_new then
            table.insert(tokens, opt.joiner)
          end
          curtok = ''
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
        elseif is_number then
          if not(number == true or space == true) then
            local addjoiner = false
            if opt.joiner_annotate then
              if opt.joiner_new then
                addjoiner = true
              else
                if not(letter) then
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
                curtok = opt.joiner
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
    if j > 1 and not tokens[j-1].rightsep and not tokens[j].leftsep then
      dline = dline .. " "
    end
    local word = tokens[j].w
    if opt.case_feature then
      word = case.restoreCase(word, tokens[j].feats)
    end
    dline = dline .. word
  end
  return dline
end

return tokenizer
