onmt = {}

require('./modules/init')

onmt.data = require('./data/init')
onmt.train = require('./train/init')
onmt.translate = require('./translate/init')
onmt.utils = require('./utils/init')

onmt.Constants = require('./Constants')
onmt.Models = require('./Models')

return onmt
