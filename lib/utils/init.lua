require('torch')

utils = {}

utils.Cuda = require('lib.utils.cuda')
utils.Dict = require('lib.utils.dict')
utils.FileReader = require('lib.utils.file_reader')
utils.Tensor = require('lib.utils.tensor')
utils.Opt = require('lib.utils.opt')
utils.Table = require('lib.utils.table')
utils.String = require('lib.utils.string')
utils.Memory = require('lib.utils.memory')
utils.Parallel = require('lib.utils.parallel')

return utils
