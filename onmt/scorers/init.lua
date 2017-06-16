local scorers = {}

scorers.bleu = require 'onmt.scorers.bleu'
scorers.gleu = require 'onmt.scorers.gleu'

return scorers
