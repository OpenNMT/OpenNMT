onmt = onmt or {}

require('onmt.modules.Sequencer')
require('onmt.modules.Encoder')
require('onmt.modules.BiEncoder')
require('onmt.modules.DBiEncoder')
require('onmt.modules.PDBiEncoder')
require('onmt.modules.Decoder')

require('onmt.modules.Network')

require('onmt.modules.Bridge')

require('onmt.modules.GRU')
require('onmt.modules.LSTM')

require('onmt.modules.MaskedSoftmax')
require('onmt.modules.WordEmbedding')
require('onmt.modules.FeaturesEmbedding')

require('onmt.modules.NoAttention')
require('onmt.modules.GlobalAttention')

require('onmt.modules.Generator')

require('onmt.modules.JoinReplicateTable')
require('onmt.modules.ParallelClassNLLCriterion')
require('onmt.modules.LayerNormalization')

return onmt
