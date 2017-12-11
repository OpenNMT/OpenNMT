local hc = require('httpclient').new()

local myopt =
{
  {
    '-pos_feature', false,
    [[Use treetagger to inject pos tags, the parameter is the path to the model to use. `treetagger`
      is expected to be in found in executable path.]]
  },
  {
    '-pos_server_host', 'localhost',
    [[POS server to use.]]
  },
  {
    '-pos_server_port', 3000,
    [[Port on the POS server to use.]]
  }
}

local function declareOptsFn(cmd)
  cmd:setCmdLineOptions(myopt, 'Tokenizer')
end

local function treetaggerFn(opt, tokens)
  if opt.pos_feature then
    local tok_nofeats = ''
    for _,v in ipairs(tokens) do
      local p = v:find('￨')
      if p then
        v = v:sub(1,p-1)
      end
      if tok_nofeats ~= '' then
        tok_nofeats = tok_nofeats..' '
      end
      tok_nofeats = tok_nofeats..v
    end
    local res = hc:post("http://"..opt.pos_server_host..':'..opt.pos_server_port..'/pos', tok_nofeats)
    assert(res.code==200)
    local s = string.gsub(res.body, "\t", "￨")
    local idx = 1
    for pos in string.gmatch(s, "%S+") do
      tokens[idx] = tokens[idx] .. '￨' .. pos
      idx = idx + 1
    end
  end
  return tokens
end

return {
  post_tokenize = treetaggerFn,
  hookName = function() return "treetagger" end,
  declareOpts = declareOptsFn
}
