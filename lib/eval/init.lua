require('torch')

eval = {}

eval.Beam = require('lib.eval.beam')
eval.PhraseTable = require('lib.eval.phrase_table')
eval.Translate = require('lib.eval.translate')

return eval
