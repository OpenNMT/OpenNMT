local mecab = require("mecab")
local parser = mecab:new("-Owakati")
local String = require ('onmt.utils.String')

local function mecab(sentence, opt)
  tokens = parser:parse(sentence)
  tokens = tokens:gsub( "%s+$", "")
  if opt.joiner_annotate then
    tokens = tokens:gsub("%s+", opt.joiner .. " ")
  end
  return String.split(tokens, " ")
end

function mytokenization(opt, tokens)
  res = {}
  for idx=1, #tokens do
    if opt.joiner_annotate then 
      if string.match(tokens[idx], opt.joiner) then
        table.insert(res, tokens[idx])
      else
        mecabTok = mecab(tokens[idx], opt)
        for k, v in pairs(mecabTok) do table.insert(res, v) end
        end
      else
        mecabTok = mecab(tokens[idx], opt)
        for k, v in pairs(mecabTok) do table.insert(res, v) end
      end
    end
    return res
end

return {
  additional_tokenize = mytokenization
}
