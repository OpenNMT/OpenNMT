local tester = ...

local tokenizerTest = torch.TestSuite()

local tokenizer = require('tools.utils.tokenizer')
local BPE = require ('tools.utils.BPE')

local cmd = onmt.utils.ExtendedCmdLine.new()
tokenizer.declareOpts(cmd)

-- check if tokenization of a is b - and if detok is set, if the detokenization results is back to original
local function testTok(opt, a, b, detok)
  local bpe
  if opt.bpe_model ~= '' then
    bpe = BPE.new(opt)
  end

  local tok = tokenizer.tokenize(opt, a, bpe)
  tester:eq(table.concat(tok, '◊'), string.gsub(b, ' ', '◊'))
  if detok == nil then return end
  tester:assert((tokenizer.detokenize(table.concat(tok, ' '), opt)==a)==detok)
end

-- check if tokenization/detokenization of a is b
local function testTokDetok(opt, a, b)
  local tok = table.concat(tokenizer.tokenize(opt, a), ' ')
  local detok = tokenizer.detokenize(tok, opt)
  tester:eq(b, detok)
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

function tokenizerTest.basicDetokenization()
  local opt = cmd:parse({'-mode','conservative'})
  testTokDetok(opt, "49th meeting Social and human rights questions: human rights [14 (g)]", "49th meeting Social and human rights questions : human rights [ 14 ( g ) ]")
  opt = cmd:parse({'-joiner_annotate'})
  testTokDetok(opt, "49th meeting Social and human rights questions: human rights [14 (g)]", "49th meeting Social and human rights questions: human rights [14 (g)]")
end

function tokenizerTest.bpebasic()
  local opt = cmd:parse({'-bpe_model','test/data/bpe-models/testcode','-joiner_annotate'})
  testTok(opt, "abcdimprovement联合国", "a￭ b￭ c￭ d￭ impr￭ ovemen￭ t￭ 联合￭ 国", true)

  opt = cmd:parse({'-bpe_model','test/data/bpe-models/fr500','-joiner_annotate','-mode','aggressive'})
  local raw = {
    [[Il n'y a encore aucun décompte de corps, c'est notre estimation, a-t-il déclaré depuis Genève.]],
    [[Le SMR mesure notamment l'efficacité et les effets indésirables d'un produit.]],
    [[Yasmina Reza retrouve le Théâtre Antoine (01.42.08.77.71) qui propose une reprise de Conversations après un enterrement, sa première pièce qui la révéla au public.]],
    [[Dans tous les cas, il y a eu une très bonne réactivité médicale.]],
    [[«J'ai reçu plusieurs offres d'emploi de très grandes entreprises», annonce-t-il, sans en dire plus.]],
    [[Le temps manquerait puisque celui-ci s'est engagé à confirmer la nomination deux jours plus tard, soit le 1er mars.]],
    [[WiMAX est avant tout une famille de normes, définissant les connexions à haut-débit par voie hertzienne.]],
    [[Nous ne faisons que de bonnes expériences avec cette façon décontractée d'accueillir nos clients.]]
  }
  local tok = {
    [[Il n ￭'￭ y a en￭ co￭ re au￭ c￭ un dé￭ com￭ p￭ te de c￭ or￭ p￭ s ￭, c ￭'￭ est no￭ tre esti￭ m￭ ation ￭, a ￭-￭ t ￭-￭ il déclar￭ é depuis G￭ en￭ è￭ ve ￭.]],
    [[Le S￭ M￭ R m￭ es￭ u￭ re no￭ ta￭ m￭ ment l ￭'￭ e￭ ffi￭ ca￭ ci￭ té et les eff￭ e￭ ts in￭ dé￭ si￭ ra￭ b￭ les d ￭'￭ un produ￭ it ￭.]],
    [[Y￭ as￭ min￭ a R￭ e￭ z￭ a re￭ tr￭ ouv￭ e le T￭ h￭ é￭ â￭ tre A￭ n￭ to￭ ine ( ￭0￭ 1 ￭. ￭4￭ 2 ￭. ￭0￭ 8 ￭. ￭7￭ 7 ￭. ￭7￭ 1 ￭) qui pro￭ po￭ se une re￭ pr￭ ise de C￭ on￭ ver￭ s￭ ations après un en￭ ter￭ r￭ ement ￭, sa premi￭ ère pi￭ è￭ ce qui la ré￭ v￭ é￭ la au publi￭ c ￭.]],
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

function tokenizerTest.segment_alphabet_change()
  local opt = cmd:parse({'-segment_alphabet_change'})
  testTok(opt, "rawБ", "raw Б")
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

return tokenizerTest
