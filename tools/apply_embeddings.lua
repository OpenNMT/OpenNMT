require('torch')
require('onmt.init')

local cmd = onmt.utils.ExtendedCmdLine.new('apply_embeddings.lua')

cmd:setCmdLineOptions(
  {
    {
      '-txt_src', '',
      [[Tokenized text file to apply embeddings on.]],
      {
        valid = onmt.utils.ExtendedCmdLine.fileExists
      }
    },
    {
      '-txt_tgt', '',
      [[Aligned target file.]],
      {
        valid = onmt.utils.ExtendedCmdLine.fileNullOrExists
      }
    },
    {
      '-dict', '',
      [[Dictionary]],
      {
        valid = onmt.utils.ExtendedCmdLine.fileExists
      }
    },
    {
      '-embed_data', '',
      [[Embedding model corresponding to dictionary generated with embeddings.lua.]],
      {
        valid = onmt.utils.ExtendedCmdLine.fileExists
      }
    },
    {
      '-save_prefix', '',
      [[Output file prefix (.src,.tgt) will be saved.]],
      {
        valid = onmt.utils.ExtendedCmdLine.nonEmpty
      }
    }
  }, 'Data')

onmt.utils.Logger.declareOpts(cmd)

local opt = cmd:parse(arg)

local function main()
  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)

  local embeddingWeights = torch.load(opt.embed_data)

  local Vocabulary = onmt.data.Vocabulary
  local dict = Vocabulary.init('source',
                               opt.txt_src,
                               opt.dict,
                               { 50000 },
                               {},
                               '',
                               function() return true end,
                               false,
                               false)

  assert(dict.words:size(1) == embeddingWeights:size(1))
  local fsrc = io.open(opt.save_prefix .. ".src", "w")
  local ftgt
  if opt.txt_tgt ~= '' then
    ftgt = io.open(opt.save_prefix .. ".tgt", "w")
  end

  local wordEmbedding = onmt.WordEmbedding.new(dict.words:size(1),
                                               embeddingWeights:size(2),
                                               embeddingWeights)

  local readerSrc = onmt.utils.FileReader.new(opt.txt_src)

  local readerTgt
  if opt.txt_tgt ~= '' then
    readerTgt = onmt.utils.FileReader.new(opt.txt_tgt)
  end

  local count = 1

  while true do
    local tokensSrc = readerSrc:next()

    if tokensSrc == nil then
      break
    end
    local IDX = 'IDX' .. count
    local words, feats = onmt.utils.Features.extract(tokensSrc)
    local vec = dict.words:convertToIdx(words, onmt.Constants.UNK_WORD)
    assert(#feats == 0)
    fsrc:write(IDX .. ' [\n')
    for i = 1, vec:size(1) do
      local we = wordEmbedding:forward(torch.LongTensor(1):fill(vec[i]))[1]
      for j = 1, embeddingWeights:size(2) do
        if j > 1 then
          fsrc:write(" ")
        end
        fsrc:write(string.format("%.4f", we[j]))
      end
      if i == vec:size(1) then
        fsrc:write(" ]")
      end
      fsrc:write("\n")
    end
    if ftgt then
      local tokensTgt = readerTgt:next()
      ftgt:write(IDX .. ' ' .. table.concat(tokensTgt, ' ') .. '\n')
    end
    count = count + 1
  end
end

main()
