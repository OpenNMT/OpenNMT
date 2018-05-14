require('onmt.init')

local path = require('pl.path')

local cmd = onmt.utils.ExtendedCmdLine.new('release_model.lua')

local options = {
  {
    '-model', '',
    [[Path to the trained model to release.]],
    {
      valid = onmt.utils.ExtendedCmdLine.fileExists
    }
  },
  {
    '-quantize', '',
    [[Quantization schema.]],
    {
      enum = { '', 'int16:1000' }
    }
  },
  {
    '-output_model', '',
    [[Path the released model. If not set, the `release` suffix will be automatically
      added to the model filename.]]
  },
  {
    '-force', false,
    [[Force output model creation even if the target file exists.]]
  }
}

cmd:setCmdLineOptions(options, 'Model')

onmt.utils.Cuda.declareOpts(cmd)
onmt.utils.Logger.declareOpts(cmd)

local opt = cmd:parse(arg)

local function isModel(object)
  return torch.type(object) == 'table' and object.modules
end

local function releaseModule(object, tensorCache)
  tensorCache = tensorCache or {}
  if object.release then
    object:release()
  end
  object:apply(function (m)
    for k, v in pairs(m) do
      if torch.isTensor(v) and v:dim()>0 then
        v:mul(1000)
        m[k] = v:short()
        m[k]:storage():resize(#m[k]:storage()+64)
      end
    end
  end)
  object:clearState()
  object:apply(function (m)
    nn.utils.clear(m, 'gradWeight', 'gradBias')
    for k, v in pairs(m) do
      if type(v) == 'function' then
        m[k] = nil
      end
    end
  end)
end

local function releaseModel(model, tensorCache)
  tensorCache = tensorCache or {}
  for _, object in pairs(model.modules) do
    if isModel(object) then
      releaseModel(object, tensorCache)
    else
      releaseModule(object, tensorCache)
    end
  end
end

local function main()
  assert(path.exists(opt.model), 'model \'' .. opt.model .. '\' does not exist.')

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level, opt.log_tag)

  if opt.output_model:len() == 0 then
    if opt.model:sub(-3) == '.t7' then
      opt.output_model = opt.model:sub(1, -4) -- copy input model without '.t7' extension
    else
      opt.output_model = opt.model
    end
    if opt.quantize == '' then
      opt.output_model = opt.output_model .. '_release.t7'
    else
      opt.output_model = opt.output_model .. '_quantize_' .. opt.quantize .. '.t7'
    end
  end

  if not opt.force then
    assert(not path.exists(opt.output_model),
           'output model already exists; use -force to overwrite.')
  end

  onmt.utils.Cuda.init(opt)

  _G.logger:info('Loading model \'' .. opt.model .. '\'...')

  local checkpoint
  local _, err = pcall(function ()
    checkpoint = torch.load(opt.model)
  end)
  if err then
    error('unable to load the model (' .. err .. '). If you are releasing a GPU model, it needs to be loaded on the GPU first (set -gpuid > 0)')
  end

  _G.logger:info('... done.')

  assert(not checkpoint.quantize, 'cannot release quantized model.')

  _G.logger:info('Converting model...')
  checkpoint.info = nil
  for _, object in pairs(checkpoint.models) do
    if isModel(object) then
      releaseModel(object)
    else
      releaseModule(object)
    end
  end
  _G.logger:info('... done.')

  if opt.quantize ~= '' then
    checkpoint.quantize = opt.quantize
  end

  _G.logger:info('Releasing model \'' .. opt.output_model .. '\'...')
  torch.save(opt.output_model, checkpoint)
  _G.logger:info('... done.')

  _G.logger:shutDown()
end

main()
