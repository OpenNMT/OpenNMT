local tester = ...

local tokenizerTest = torch.TestSuite()

local tokenizer = require('tools.utils.tokenizer')
local BPE = require ('tools.utils.BPE')

local cmd = onmt.utils.ExtendedCmdLine.new()
tokenizer.declareOpts(cmd)

-- insert on the fly the option depending if there is a hook selected
onmt.utils.HookManager.updateOpt(arg, cmd)
onmt.utils.HookManager.declareOpts(cmd)

-- check if tokenization of a is b - and if detok is set, if the detokenization results is back to original
local function testTok(opt, a, b, detok)
  local bpe
  if opt.bpe_model ~= '' then
    bpe = BPE.new(opt)
  end

  local tok = tokenizer.tokenize(opt, a, bpe)
  tester:eq(table.concat(tok, ' '), string.gsub(b, ' ', ' '))
  if not detok then return end
  tester:assert((tokenizer.detokenizeLine(opt, table.concat(tok, ' '))==a)==detok)
end

-- check if tokenization/detokenization of a is b
local function testTokDetok(opt, a, b)
  local tok = table.concat(tokenizer.tokenize(opt, a), ' ')
  local detok = tokenizer.detokenizeLine(opt, tok)
  tester:eq(b, detok)
end

-- check if detokenization of a is b
local function testDetok(opt, a, b)
  tester:eq(b, tokenizer.detokenizeLine(opt, a))
end

function tokenizerTest.basic()
  local opt = cmd:parse('')
  testTok(opt, "Your Hardware-Enablement Stack (HWE) is supported until April 2019.", "Your Hardware-Enablement Stack ( HWE ) is supported until April 2019 .", false)
  opt = cmd:parse({'-joiner_annotate'})
  testTok(opt, "Your Hardware-Enablement Stack (HWE) is supported until April 2019.", "Your Hardware-Enablement Stack (￭ HWE ￭) is supported until April 2019 ￭.", true)
  opt = cmd:parse({'-joiner_annotate','-mode', 'aggressive'})
  testTok(opt, "Isn't it so-greatly working?", "Isn ￭'￭ t it so ￭-￭ greatly working ￭?", true)
  testTok(opt, "MP3", "MP ￭3", true)
end

function tokenizerTest.combiningMark()
  local opt = cmd:parse({'-mode','conservative','-joiner_annotate'})
  testTok(opt, "वर्तमान लिपि (स्क्रिप्ट) खो जाएगी।", "वर्तमान लिपि (￭ स्क्रिप्ट ￭) खो जाएगी ￭।", true)
end

function tokenizerTest.spaceTokenization()
  local opt = cmd:parse({'-mode','space'})
  testTok(opt, "49th meeting Social and human rights questions: human rights [14 (g)]", "49th meeting Social and human rights questions: human rights [14 (g)]")
  opt = cmd:parse({'-mode','space', '-case_feature'})
  testTok(opt, "49th meeting Social and human rights questions: human rights [14 (g)]", "49th￨L meeting￨L social￨C and￨L human￨L rights￨L questions:￨L human￨L rights￨L [14￨N (g)]￨L")
end

function tokenizerTest.protectedSequence()
  local opt = cmd:parse({'-mode','conservative','-joiner_annotate'})
  testTok(opt, "｟1,023｠km", "｟1,023｠￭ km", true)
  testTok(opt, "A｟380｠", "A ￭｟380｠", true)
  testTok(opt, "｟1,023｠｟380｠", "｟1,023｠￭ ｟380｠", true)
  testTok(opt, "｟1023｠.", "｟1023｠ ￭.", true)
  testTok(opt, "$｟0.23｠", "$￭ ｟0.23｠", true)
  testTok(opt, "｟0.23｠$", "｟0.23｠ ￭$", true)
  testTok(opt, "｟US$｠23", "｟US$｠￭ 23", true)
  testTok(opt, "1｟ABCD｠0", "1 ￭｟ABCD｠￭ 0", true)
  testTok(opt, "$1", "$￭ 1", true)
  opt = cmd:parse({'-mode','conservative','-joiner_annotate','-joiner_new'})
  testTok(opt, "｟1,023｠km", "｟1,023｠ ￭ km", true)
  testTok(opt, "A｟380｠", "A ￭ ｟380｠", true)
  testTok(opt, "｟1,023｠｟380｠", "｟1,023｠ ￭ ｟380｠", true)
  testTok(opt, "｟1023｠.", "｟1023｠ ￭ .", true)
end

function tokenizerTest.protectedSequenceAndCaseFeature()
  local opt = cmd:parse({'-mode', 'conservative', '-case_feature', '-joiner_annotate'})
  testTok(opt, "｟AbC｠", "｟AbC｠￨N", true)
  testTok(opt, "｟AbC｠.", "｟AbC｠￨N ￭.￨N", true)
  testTok(opt, "Abc｟DeF｠.", "abc￨C ￭｟DeF｠￨N ￭.￨N", true)
  testTok(opt, "Abc｟DeF｠ghi", "abc￨C ￭｟DeF｠￭￨N ghi￨L", true)
  testDetok(opt, "｟abc｠￨U", "｟abc｠")
  opt = cmd:parse({'-mode', 'conservative', '-case_feature', '-segment_case'})
  testTok(opt, "｟WiFi｠", "｟WiFi｠￨N", true)
end

function tokenizerTest.aggressive()
  local opt = cmd:parse({'-mode','aggressive','-joiner_annotate'})
  testTok(opt, "｟1,023｠km", "｟1,023｠￭ km", true)
  testTok(opt, "A｟380｠", "A ￭｟380｠", true)
  testTok(opt, "｟1,023｠｟380｠", "｟1,023｠￭ ｟380｠", true)
  testTok(opt, "｟1023｠.", "｟1023｠ ￭.", true)
  testTok(opt, "$｟0.23｠", "$￭ ｟0.23｠", true)
  testTok(opt, "｟0.23｠$", "｟0.23｠ ￭$", true)
  testTok(opt, "｟US$｠23", "｟US$｠￭ 23", true)
  testTok(opt, "1｟ABCD｠0", "1 ￭｟ABCD｠￭ 0", true)
  testTok(opt, "$1", "$￭ 1", true)
  testTok(opt, "A380", "A ￭380", true)
end

function tokenizerTest.basicDetokenization()
  local opt = cmd:parse({'-mode','conservative'})
  testTokDetok(opt, "49th meeting Social and human rights questions: human rights [14 (g)]", "49th meeting Social and human rights questions : human rights [ 14 ( g ) ]")
  opt = cmd:parse({'-joiner_annotate'})
  testTokDetok(opt, "49th meeting Social and human rights questions: human rights [14 (g)]", "49th meeting Social and human rights questions: human rights [14 (g)]")
end

function tokenizerTest.detokenizeTable()
  local words = {
    '<bpt i="1"><ChrStyle name="bold"><bpt/>',
    'a', '￭4￭', 'r', '<ept i="1"></ChrStyle><ept/>'
  }
  local features = { { 'N', 'C', 'N', 'C', 'N' } }

  tester:eq(tokenizer.detokenize({joiner='￭', case_feature=true}, words, features),
            '<bpt i="1"><ChrStyle name="bold"><bpt/> A4R <ept i="1"></ChrStyle><ept/>')
  tester:eq(tokenizer.detokenize({joiner='￭'}, words),
            '<bpt i="1"><ChrStyle name="bold"><bpt/> a4r <ept i="1"></ChrStyle><ept/>')
end

function tokenizerTest.bpebasic()
  local opt = cmd:parse({'-bpe_model','test/data/bpe-models/testcode','-joiner_annotate'})
  testTok(opt, "abcdimprovement联合国", "a￭ b￭ c￭ d￭ impr￭ ovemen￭ t￭ 联合￭ 国", true)

  opt = cmd:parse({'-bpe_model','test/data/bpe-models/fr500','-joiner_annotate','-mode','aggressive'})
  local raw = {
    [[Il n'y a encore aucun décompte de corps, c'est notre estimation, a-t-il déclaré ｟depuis Genève｠.]],
    [[Le SMR mesure notamment l'efficacité et les effets indésirables d'un produit.]],
    [[Yasmina Reza retrouve le Théâtre Antoine (01.42.08.77.71) qui propose une reprise de Conversations après un enterrement, sa première pièce qui la révéla au public.]],
    [[Dans tous les cas, il y a eu une très bonne réactivité médicale.]],
    [[«J'ai reçu plusieurs offres d'emploi de très grandes entreprises», annonce-t-il, sans en dire plus.]],
    [[Le temps manquerait puisque celui-ci s'est engagé à confirmer la nomination deux jours plus tard, soit le 1er mars.]],
    [[WiMAX est avant tout une famille de normes, définissant les connexions à haut-débit par voie hertzienne.]],
    [[Nous ne faisons que de bonnes expériences avec cette façon décontractée d'accueillir nos clients.]]
  }
  local tok = {
    [[Il n ￭'￭ y a en￭ co￭ re au￭ c￭ un dé￭ com￭ p￭ te de c￭ or￭ p￭ s ￭, c ￭'￭ est no￭ tre esti￭ m￭ ation ￭, a ￭-￭ t ￭-￭ il déclar￭ é ｟depuis％0020Genève｠ ￭.]],
    [[Le S￭ M￭ R m￭ es￭ u￭ re no￭ ta￭ m￭ ment l ￭'￭ e￭ ffi￭ ca￭ ci￭ té et les eff￭ e￭ ts in￭ dé￭ si￭ ra￭ b￭ les d ￭'￭ un produ￭ it ￭.]],
    [[Y￭ as￭ min￭ a R￭ e￭ z￭ a re￭ tr￭ ouv￭ e le T￭ h￭ é￭ â￭ tre A￭ n￭ to￭ ine (￭ 0￭ 1 ￭.￭ 4￭ 2 ￭.￭ 0￭ 8 ￭.￭ 7￭ 7 ￭.￭ 7￭ 1 ￭) qui pro￭ po￭ se une re￭ pr￭ ise de C￭ on￭ ver￭ s￭ ations après un en￭ ter￭ r￭ ement ￭, sa premi￭ ère pi￭ è￭ ce qui la ré￭ v￭ é￭ la au publi￭ c ￭.]],
    [[Dans tou￭ s les ca￭ s ￭, il y a eu une tr￭ ès b￭ on￭ ne ré￭ ac￭ ti￭ vi￭ té mé￭ di￭ ca￭ le ￭.]],
    [[«￭ J ￭'￭ ai re￭ ç￭ u pl￭ usi￭ eurs o￭ ff￭ res d ￭'￭ em￭ pl￭ o￭ i de tr￭ ès gran￭ des entr￭ e￭ pr￭ is￭ es ￭» ￭, ann￭ on￭ ce ￭-￭ t ￭-￭ il ￭, sans en di￭ re plus ￭.]],
    [[Le t￭ emp￭ s man￭ qu￭ er￭ ait pu￭ is￭ que c￭ el￭ u￭ i ￭-￭ c￭ i s ￭'￭ est en￭ g￭ ag￭ é à con￭ fi￭ r￭ m￭ er la nom￭ in￭ ation deux jou￭ rs plus t￭ ar￭ d ￭, s￭ oit le 1￭ er mar￭ s ￭.]],
    [[W￭ i￭ M￭ A￭ X est av￭ ant tout une f￭ am￭ ille de n￭ or￭ mes ￭, dé￭ fin￭ iss￭ ant les con￭ n￭ ex￭ i￭ ons à h￭ au￭ t ￭-￭ dé￭ bi￭ t par v￭ oi￭ e h￭ er￭ t￭ z￭ i￭ enne ￭.]],
    [[N￭ ous ne f￭ ai￭ s￭ ons que de b￭ onn￭ es ex￭ p￭ éri￭ en￭ ces avec cette f￭ a￭ ç￭ on dé￭ contr￭ ac￭ t￭ ée d ￭'￭ ac￭ cu￭ e￭ illi￭ r no￭ s c￭ li￭ ents ￭.]]
  }
  for i = 1, #raw do
    testTok(opt, raw[i], tok[i], true)
  end
end

function tokenizerTest.bpeModePrefix()
  local opt = cmd:parse({'-bpe_model','test/data/bpe-models/codes_prefix.fr','-mode','aggressive'})
  testTok(opt, "Seulement seulement il vais nonseulement seulementnon à Verdun", "S e u lement seulement il v ais n on se u lement seulement n on à V er d un", false)
end

function tokenizerTest.bpeModeNofix()
  local opt = cmd:parse({'-bpe_model','test/data/bpe-models/codes_nofix.fr','-mode','aggressive'})
  testTok(opt, "Seulement seulement il vais nonseulement seulementnon à Verdun", "S e u lement seulement il v ais n on seulement seulement n on à V er d un", false)
end

function tokenizerTest.bpeModeBothfix()
  local opt = cmd:parse({'-bpe_model','test/data/bpe-models/codes_bothfix.fr','-mode','aggressive'})
  testTok(opt, "Seulement seulement il vais nonseulement seulementnon à Verdun", "S eu lement seulement il va is n on s eu lement seu l emen t n on à V er du n", false)
end

function tokenizerTest.bpeCaseInsensitive()
  local opt = cmd:parse({'-bpe_model','test/data/bpe-models/codes_suffix_case_insensitive.fr','-mode','aggressive'})
  testTok(opt, "Seulement seulement il vais nonseulement seulementnon à Verdun", "Seulement seulement il va is n on seulement seu l em ent n on à Ver d un", false)
end

function tokenizerTest.test_substitute()
  local opt = cmd:parse({'-mode','conservative'})
  testTok(opt, [[test￭ protect￨, ：, and ％ or ＃...]],
               [[test ■ protect │ , : , and % or # . . .]], false)
end

function tokenizerTest.case_feature()
  local opt = cmd:parse({'-mode','conservative','-joiner_annotate', '-case_feature'})
  testTok(opt, [[test \\\\a Capitalized lowercased UPPERCASÉ miXêd - cyrillic-Б]],
               [[test￨L \￨N ￭\￨N ￭\￨N ￭\￭￨N a￨L capitalized￨C lowercased￨L uppercasé￨U mixêd￨M -￨N cyrillic-б￨M]], false)
end

function tokenizerTest.segment_case()
  local opt = cmd:parse({'-mode','conservative','-joiner_annotate', '-segment_case', '-case_feature'})
  testTok(opt, [[WiFi]],
               [[wi￭￨C fi￨C]], true)
end

function tokenizerTest.segment_alphabet()
  local opt = cmd:parse({'-segment_alphabet','Han','-joiner_annotate'})
  testTok(opt, "rawБ", "rawБ", true)
  local raw = "有入聲嘅唐話往往有陽入對轉，即係入聲韻尾同鼻音韻尾可以轉化。比如粵語嘅「抌」（dam）「揼」（dap），意思接近，意味微妙，區別在於-m同-p嘅轉換。"
  local tok = "有￭ 入￭ 聲￭ 嘅￭ 唐￭ 話￭ 往￭ 往￭ 有￭ 陽￭ 入￭ 對￭ 轉 ￭，￭ 即￭ 係￭ 入￭ 聲￭ 韻￭ 尾￭ 同￭ 鼻￭ 音￭ 韻￭ 尾￭ 可￭ 以￭ 轉￭ 化 ￭。￭ 比￭ "..
              "如￭ 粵￭ 語￭ 嘅 ￭「￭ 抌 ￭」 ￭（￭ dam ￭） ￭「￭ 揼 ￭」 ￭（￭ dap ￭） ￭，￭ 意￭ 思￭ 接￭ 近 ￭，￭ 意￭ 味￭ 微￭ 妙 ￭，￭ 區￭ 別￭ 在￭ 於-m同-p嘅￭ 轉￭ 換 ￭。"
  testTok(opt, raw, tok)
end

function tokenizerTest.segment_numbers()
  local opt = cmd:parse({'-segment_numbers','-joiner_annotate','-mode','aggressive'})
  testTok(opt, "1984 mille neuf cent quatrevingt-quatre", "1￭ 9￭ 8￭ 4 mille neuf cent quatrevingt ￭-￭ quatre", true)
end

function tokenizerTest.segment_alphabet_change()
  local opt = cmd:parse({'-segment_alphabet_change'})
  testTok(opt, "rawБ", "raw Б")
end

function tokenizerTest.placeholder_joiners()
  local opt = cmd:parse({'-mode','conservative','-joiner_annotate'})
  testTok(opt, "｟ph｠abc", "｟ph｠￭ abc", false)
  testTok(opt, "｟ph｠123", "｟ph｠￭ 123", false)
  testTok(opt, "｟ph｠.", "｟ph｠ ￭.", false)
  testTok(opt, "abc｟ph｠", "abc ￭｟ph｠", false)
  testTok(opt, "123｟ph｠", "123 ￭｟ph｠", false)
  testTok(opt, "-｟ph｠", "-￭ ｟ph｠", false)
  testTok(opt, "｟ph｠｟ph｠", "｟ph｠￭ ｟ph｠", false)
end

function tokenizerTest.bugCTok56()
  local opt = cmd:parse({'-mode','conservative','-joiner_annotate'})
  testTok(opt, "61.\tSet", "61 ￭. Set")
  opt.joiner_annotate = false
  testTok(opt, "￭￨％＃：", "■ │ % # :", false)
end

function tokenizerTest.real()
  local opt = cmd:parse({'-mode','conservative','-joiner_annotate'})
  local raw = {
    [[Reste à savoir si une nouvelle initiative arabe ou une impulsion Berry-Siniora pourraient changer quoi que ce soit.]],
    [[Le programme d'investissement dans la fabrication de neige de culture est en augmentation de 50% cette saison en Haute-Savoie.]],
    [[06H28 locales (21H28 GMT mardi), depuis le centre spatial d'Uchinoura (sud du Japon), a précisé l'Agence.]],
    [[Le chikungunya provoque des douleurs articulaires terribles, mais c'est à croire qu'il a aussi paralysé le gouvernement français.]],
    [[La majorité de la population pratiquent le riz sur brûlis (tavy).]],
    [[Panama, la moitié du nom de son inventeur, Panamarenko, à qui l'institution belge consacre une rétrospective.]],
    [[«Je reçois mes premiers mails à 7 heures du matin, les derniers à 23 heures», reconnaît-il.]],
    [[V.D. Je me souviens par exemple de Quelqu'un va venir de Jon Fosse (en 1999, ndlr).]],
    [[Le contexte politique colombien ne se prête à aucune tractation entre le gouvernement et les Farc.]]
  }
  local tok = {
    [[Reste à savoir si une nouvelle initiative arabe ou une impulsion Berry-Siniora pourraient changer quoi que ce soit ￭.]],
    [[Le programme d ￭'￭ investissement dans la fabrication de neige de culture est en augmentation de 50 ￭% cette saison en Haute-Savoie ￭.]],
    [[06H28 locales (￭ 21H28 GMT mardi ￭) ￭, depuis le centre spatial d ￭'￭ Uchinoura (￭ sud du Japon ￭) ￭, a précisé l ￭'￭ Agence ￭.]],
    [[Le chikungunya provoque des douleurs articulaires terribles ￭, mais c ￭'￭ est à croire qu ￭'￭ il a aussi paralysé le gouvernement français ￭.]],
    [[La majorité de la population pratiquent le riz sur brûlis (￭ tavy ￭) ￭.]],
    [[Panama ￭, la moitié du nom de son inventeur ￭, Panamarenko ￭, à qui l ￭'￭ institution belge consacre une rétrospective ￭.]],
    [[«￭ Je reçois mes premiers mails à 7 heures du matin ￭, les derniers à 23 heures ￭» ￭, reconnaît-il ￭.]],
    [[V.D ￭. Je me souviens par exemple de Quelqu ￭'￭ un va venir de Jon Fosse (￭ en 1999 ￭, ndlr ￭) ￭.]],
    [[Le contexte politique colombien ne se prête à aucune tractation entre le gouvernement et les Farc ￭.]]
  }
  for i = 1, #raw do
    testTok(opt, raw[i], tok[i], true)
  end
end

function tokenizerTest.hooks()
  local hookName = _G.hookManager:call("hookName")
  if hookName == "chartok" then
    local opt = cmd:parse({'-mode','char'})
    testTok(opt, "49th meeting Social and human rights questions [14 (g)]",
                 "4 9 t h ▁ m e e t i n g ▁ S o c i a l ▁ a n d ▁ h u m a n ▁ r i g h t s ▁ q u e s t i o n s ▁ [ 1 4 ▁ ( g ) ]")
  elseif hookName == "sentencepiece" then
    local opt = cmd:parse({'-mode','none', '-sentencepiece' ,'test/data/sample.model'})
    testTok(opt, "une impulsion Berry-Siniora pourraient changer quoi",
                 "▁une ▁impu l sion ▁B erry - S ini or a ▁pourraient ▁change r ▁quoi")
  end
end

return tokenizerTest
