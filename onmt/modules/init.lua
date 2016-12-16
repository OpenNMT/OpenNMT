onmt = onmt or {}

require('./Sequencer')
require('./Encoder')
require('./BiEncoder')
require('./Decoder')

require('./LSTM')

require('./MaskedSoftmax')
require('./WordEmbedding')
require('./FeaturesEmbedding')
require('./GlobalAttention')

require('./Generator')
require('./FeaturesGenerator')

return onmt
