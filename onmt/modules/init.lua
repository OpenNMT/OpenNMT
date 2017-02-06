onmt = onmt or {}

require('onmt.modules.Sequencer')
require('onmt.modules.Encoder')
require('onmt.modules.BiEncoder')
require('onmt.modules.Decoder')

require('onmt.modules.Network')

require('onmt.modules.GRU')
require('onmt.modules.LSTM')

require('onmt.modules.MaskedSoftmax')
require('onmt.modules.WordEmbedding')
require('onmt.modules.FeaturesEmbedding')
require('onmt.modules.GlobalAttention')

require('onmt.modules.Generator')
require('onmt.modules.FeaturesGenerator')

require('onmt.modules.Criterion')

onmt.modules = { Factory = require('onmt.modules.Factory') }

return onmt
