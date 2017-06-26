local scorers = {}

scorers.bleu = require 'onmt.scorers.bleu'
scorers.dlratio = require 'onmt.scorers.dlratio'

scorers.list = { 'bleu', 'dlratio' }

return scorers
