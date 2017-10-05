require('onmt.init')

local path = require('pl.path')

local cmd = torch.CmdLine()
cmd:option('-model', '', 'trained model file')
cmd:option('-output_dir', '.', 'output directory')
cmd:option('-gpuid', 0, [[1-based identifier of the GPU to use. CPU is used when the option is < 1]])
local opt = cmd:parse(arg)

local function write_embeddings(filename, dict, embeddings)
  local file = assert(io.open(filename, 'w'))

  for i = 1, embeddings:size(1) do
    local str = dict.idxToLabel[i]
    for j = 1,  embeddings:size(2) do
      str = str .. string.format(" %5f", embeddings[i][j])
    end
    file:write(str  .. '\n')
  end
  file:close()
end

local function main()
  assert(path.exists(opt.model), 'model \'' .. opt.model .. '\' does not exist.')

  if opt.gpuid > 0 then
    require('cutorch')
    cutorch.setDevice(opt.gpuid)
  end

  print('Loading model \'' .. opt.model .. '\'...')

  local checkpoint
  local _, err = pcall(function ()
    checkpoint = torch.load(opt.model)
  end)
  if err then
    error('unable to load the model (' .. err .. '). If you are extracting a GPU model, it needs to be loaded on the GPU first (set -gpuid > 0)')
  end
  local dicts = checkpoint.dicts
  local encoder = onmt.Factory.loadEncoder(checkpoint.models.encoder)
  local decoder
  if checkpoint.models.decoder then
    decoder = onmt.Factory.loadDecoder(checkpoint.models.decoder)
  end
  local encoder_embeddings, decoder_embeddings

  encoder:apply(function(m)
      if torch.type(m) == "onmt.WordEmbedding" then
        print("Found source embeddings of size " ..  m.net.weight:size(1))
        if m.net.weight:size(1) == dicts.src.words:size() then
          encoder_embeddings = m.net.weight
        end
        return
      end
  end)

  if decoder then
    decoder:apply(function(m)
        if torch.type(m) == "onmt.WordEmbedding" then
          print("Found target embeddings of size " ..  m.net.weight:size(1))
          if m.net.weight:size(1) == dicts.tgt.words:size() then
            decoder_embeddings = m.net.weight
          end
          return
        end
    end)
  end

  print("Writing source embeddings")
  write_embeddings(opt.output_dir .. "/src_embeddings.txt", dicts.src.words, encoder_embeddings)

  if checkpoint.models.decoder then
    print("Writing target embeddings")
    write_embeddings(opt.output_dir .. "/tgt_embeddings.txt", dicts.tgt.words, decoder_embeddings)
  end
  print('... done.')
end

main()
