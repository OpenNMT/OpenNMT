onmt = {}

require('onmt.modules.init')

onmt.data = require('onmt.data.init')
onmt.train = require('onmt.train.init')
onmt.translate = require('onmt.translate.init')
onmt.utils = require('onmt.utils.init')

onmt.Constants = require('onmt.constants')
onmt.Models = require('onmt.models')

return onmt
