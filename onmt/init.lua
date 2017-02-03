onmt = {}

-- this needs to be loaded first
onmt.ExtendedCmdLine = require('onmt.utils.ExtendedCmdLine')

require('onmt.modules.init')

onmt.utils = require('onmt.utils.init')
onmt.data = require('onmt.data.init')
onmt.train = require('onmt.train.init')
onmt.translate = require('onmt.translate.init')

onmt.Constants = require('onmt.Constants')
onmt.Models = require('onmt.Models')

onmt.models = require('onmt.models.init')

onmt.Trainer = require('onmt.Trainer')

return onmt
