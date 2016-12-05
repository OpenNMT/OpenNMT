require('torch')

translate = {}

translate.Beam = require('lib.translate.beam')
translate.PhraseTable = require('lib.translate.phrase_table')
translate.Translator = require('lib.translate.translator')

return translate
