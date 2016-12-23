package = "opennmt"
version = "scm-1"

source = {
   url = "git://github.com/opennmt/opennmt",
   tag = "master"
}

description = {
   summary = "Neural Machine Translation for Torch",
   homepage = "https://github.com/opennmt/opennmt",
   license = "MIT"
}

dependencies = {
   "nn >= 1.0",
   "nngraph",
   "tds",
   "threads",
   "torch >= 7.0"
}


build = {
  type = "builtin",
  install = {
    bin = {
      onmt_preprocess = "preprocess.lua",
      onmt_train = "train.lua",
      onmt_translate = "translate.lua",
    }
  },
  
  modules = {
    onmt = "onmt/init.lua",
    ["onmt.modules.init"] = "onmt/modules/init.lua",
    ["onmt.modules.BiEncoder"] = "onmt/modules/BiEncoder.lua",
    ["onmt.modules.Decoder"] = "onmt/modules/Decoder.lua",
    ["onmt.modules.Encoder"] = "onmt/modules/Encoder.lua",
    ["onmt.modules.FeaturesEmbedding"] = "onmt/modules/FeaturesEmbedding.lua",
    ["onmt.modules.FeaturesGenerator"] = "onmt/modules/FeaturesGenerator.lua",
    ["onmt.modules.Generator"] = "onmt/modules/Generator.lua",
    ["onmt.modules.GlobalAttention"] = "onmt/modules/GlobalAttention.lua",
    ["onmt.modules.LSTM"] = "onmt/modules/LSTM.lua",
    ["onmt.modules.MaskedSoftmax"] = "onmt/modules/MaskedSoftmax.lua",
    ["onmt.modules.Sequencer"] = "onmt/modules/Sequencer.lua",
    ["onmt.modules.WordEmbedding"] = "onmt/modules/WordEmbedding.lua",
    ["onmt.data.init"] = "onmt/data/init.lua",
    ["onmt.data.Batch"] = "onmt/data/Batch.lua",
    ["onmt.data.Dataset"] = "onmt/data/Dataset.lua",
    ["onmt.train.init"] = "onmt/train/init.lua",
    ["onmt.train.Checkpoint"] = "onmt/train/Checkpoint.lua",
    ["onmt.train.EpochState"] = "onmt/train/EpochState.lua",
    ["onmt.train.Optim"] = "onmt/train/Optim.lua",
    ["onmt.translate.init"] = "onmt/translate/init.lua",
    ["onmt.translate.Beam"] = "onmt/translate/Beam.lua",
    ["onmt.translate.Translator"] = "onmt/translate/Translator.lua",
    ["onmt.translate.PhraseTable"] = "onmt/translate/PhraseTable.lua",
    ["onmt.utils.init"] = "onmt/utils/init.lua",
    ["onmt.utils.Cuda"] = "onmt/utils/Cuda.lua",
    ["onmt.utils.Dict"] = "onmt/utils/Dict.lua",
    ["onmt.utils.Features"] = "onmt/utils/Features.lua",
    ["onmt.utils.FileReader"] = "onmt/utils/FileReader.lua",
    ["onmt.utils.Log"] = "onmt/utils/Log.lua",
    ["onmt.utils.Logger"] = "onmt/utils/Logger.lua",
    ["onmt.utils.Memory"] = "onmt/utils/Memory.lua",
    ["onmt.utils.Opt"] = "onmt/utils/Opt.lua",
    ["onmt.utils.Parallel"] = "onmt/utils/Parallel.lua",
    ["onmt.utils.String"] = "onmt/utils/String.lua",
    ["onmt.utils.Table"] = "onmt/utils/Table.lua",
    ["onmt.utils.Tensor"] = "onmt/utils/Tensor.lua",
    ["onmt.Constants"] = "onmt/Constants.lua",
    ["onmt.Models"] = "onmt/Models.lua"
  }
}
