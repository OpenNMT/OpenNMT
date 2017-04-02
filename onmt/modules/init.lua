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

-- Attention modules
require('onmt.modules.GlobalAttention')
require('onmt.modules.ContextGateAttention')


require('onmt.modules.Generator')
require('onmt.modules.FeaturesGenerator')

require('onmt.modules.ParallelClassNLLCriterion')


-- Coverage modules
require('onmt.modules.Coverage.ContextCoverage')
require('onmt.modules.Coverage.CoverageAttention')

require('onmt.modules.Utils.SequenceLinear')
require('onmt.modules.Utils.Replicator')
require('onmt.modules.Utils.Energy')
require('onmt.modules.Utils.LayerNormalization')

return onmt
