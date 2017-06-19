local scorers = {}

scorers.bleu = require 'onmt.scorers.bleu'

scorers.list = { 'bleu' }

return scorers
