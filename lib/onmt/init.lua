--[[ Package
--]]


require('torch')

onmt = {}

require('lib.onmt.Sequencer')
require('lib.onmt.Encoder')
require('lib.onmt.BiEncoder')
require('lib.onmt.Decoder')

require('lib.onmt.LSTM')

require('lib.onmt.MaskedSoftmax')
require('lib.onmt.WordEmbedding')
require('lib.onmt.FeaturesEmbedding')
require('lib.onmt.GlobalAttention')

return onmt
