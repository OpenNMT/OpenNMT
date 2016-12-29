require('onmt.init')
require('tds')
local cmd = torch.CmdLine()

cmd:text("")
cmd:text("**train.lua**")
cmd:text("")

cmd:option('-config', '', [[Read options from this file]])

onmt.data.Dataset.declareOpts(cmd)

-- Options for model construction
onmt.Models.declareOpts(cmd)

-- Options for training
onmt.train.Train.declareOpts(cmd)

onmt.train.Checkpoint.declareOpts(cmd)

cmd:text("")
cmd:text("**Other options**")
cmd:text("")

onmt.utils.Cuda.declareOpts(cmd)
onmt.utils.Parallel.declareOpts(cmd)

cmd:option('-seed', 3435, [[Seed for random initialization]])
cmd:option('-json_log', false, [[Outputs logs in JSON format.]])

local opt = cmd:parse(arg)


local function main()
  local requiredOptions = {
    "data",
    "save_model"
  }

  onmt.utils.Opt.init(opt, requiredOptions)
  onmt.utils.Cuda.init(opt)
  onmt.utils.Parallel.init(opt)
  onmt.train.Train.init(opt)

  -- Setup the checkpoint.
  local checkpoint = onmt.train.Checkpoint.init(opt)

  -- Load the data.
  local dataset, trainData, validData = onmt.data.Dataset.load(opt)

  -- Build/load the model (possibly in parallel)
  if not opt.json_log then
    print('Building model...')
  end
  local model
  onmt.utils.Parallel.launch(function(idx)
    _G.model = {}
    if checkpoint.models then
      _G.model.encoder = onmt.Models.loadEncoder(checkpoint.models.encoder, idx > 1)
      _G.model.decoder = onmt.Models.loadDecoder(checkpoint.models.decoder, idx > 1)
    else
      local verbose = idx == 1 and not opt.json_log
      _G.model.encoder = onmt.Models.buildEncoder(opt, dataset.dicts.src)
      _G.model.decoder = onmt.Models.buildDecoder(opt, dataset.dicts.tgt, verbose)
    end

    for _, mod in pairs(_G.model) do
      onmt.utils.Cuda.convert(mod)
    end

    return idx, _G.model
  end, function(idx, themodel)
    if idx == 1 then
      model = themodel
    end
  end)

  local function criterionBuilder(data)
    return onmt.Models.buildCriterion(data.dicts.tgt.words:size(),
                                      data.dicts.tgt.features)
  end

  -- Train the model.
  onmt.train.Train.trainModel(model, trainData, validData, dataset, checkpoint.info,
                              criterionBuilder)
end

main()
