require('torch')

onmt = {}

onmt.utils = require('onmt.utils.init')

require('onmt.modules.init')

onmt.data = require('onmt.data.init')
onmt.train = require('onmt.train.init')
onmt.translate = require('onmt.translate.init')
onmt.tagger = require('onmt.tagger.init')
onmt.lm = require('onmt.lm.init')

onmt.Constants = require('onmt.Constants')
onmt.Factory = require('onmt.Factory')
onmt.Model = require('onmt.Model')
onmt.Seq2Seq = require('onmt.Seq2Seq')
onmt.LanguageModel = require('onmt.LanguageModel')
onmt.SeqTagger = require('onmt.SeqTagger')
onmt.ModelSelector = require('onmt.ModelSelector')

return onmt
