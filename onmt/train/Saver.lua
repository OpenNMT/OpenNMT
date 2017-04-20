-- Class for saving and loading models during training.
local Saver = torch.class('Saver')

local options = {
  {
    '-save_model', '',
    [[Model filename (the model will be saved as `<save_model>_epochN_PPL.t7`
      where `PPL` is the validation perplexity.]],
    {
      valid = onmt.utils.ExtendedCmdLine.nonEmpty
    }
  },
  {
    '-train_from', '',
    [[Path to a checkpoint.]],
    {
      valid = onmt.utils.ExtendedCmdLine.fileNullOrExists
    }
  },
  {
    '-continue', false,
    [[If set, continue the training where it left off.]]
  }
}

function Saver.declareOpts(cmd)
  cmd:setCmdLineOptions(options, 'Saver')
end

--[[ Determine from the options if a checkpoint is set. ]]
function Saver.checkpointDefined(opt)
  return opt.train_from:len() > 0
end

--[[ Load and check checkpoint information and return new options and parameters. ]]
function Saver.loadCheckpoint(opt)
  _G.logger:info('Loading checkpoint \'' .. opt.train_from .. '\'...')

  local checkpoint = torch.load(opt.train_from)

  local function restoreOption(name)
    if checkpoint.options[name] ~= nil then
      opt[name] = checkpoint.options[name]
    end
  end

  local function optionChanged(name)
    return not opt._is_default[name] and opt[name] ~= checkpoint.options[name]
  end

  -- Check if we should reuse RNG states.
  if checkpoint.info.rngStates and optionChanged('seed') then
    _G.logger:warning('New \'-seed\' value: ignoring saved RNG states.')
    checkpoint.info.rngStates = nil
  end

  -- Check if the continue mode can be applied.
  if opt.continue and (optionChanged('learning_rate') or optionChanged('start_epoch')) then
    _G.logger:warning('\'-continue\' option is used but -learning_rate or -start_epoch are set and different than previous epoch. Ignoring \'-continue\'.')
    opt.continue = false
  end

  -- Reload and check options.
  local paramChanges = {}

  for k, v in pairs(opt) do
    if k:sub(1, 1) ~= '_' then

      if opt.continue and opt._train_state[k] then
        -- Training states should be retrieved when continuing a training.
        restoreOption(k)
      elseif opt._structural[k] or opt._init_only[k] then

        -- If an option was set by the user, check that we can actually change it.
        if optionChanged(k) then
          if opt._init_only[k] then
            _G.logger:warning('Cannot change initialization option -%s. Ignoring.', k)
            restoreOption(k)
          elseif opt._structural[k] and opt._structural[k] == 0 then
            _G.logger:warning('Cannot change dynamically option -%s. Ignoring.', k)
            restoreOption(k)
          elseif opt._structural[k] and opt._structural[k] == 1 then
            paramChanges[k] = v
          end
        else
          restoreOption(k)
        end

      end

    end
  end

  if opt.continue then
    -- When continuing, some options are initialized with their last known value.
    opt.learning_rate = checkpoint.info.learningRate
    opt.start_epoch = checkpoint.info.epoch
    opt.start_iteration = checkpoint.info.iteration

    _G.logger:info('Resuming training from epoch ' .. opt.start_epoch
                     .. ' at iteration ' .. opt.start_iteration .. '...')
  else
    -- Otherwise, we can drop previous training information.
    local rngStates = checkpoint.info.rngStates
    checkpoint.info = {}
    checkpoint.info.rngStates = rngStates
  end

  return checkpoint, opt, paramChanges
end

--[[ Create a new saver holding static data to store in each model. ]]
function Saver:__init(args, model, optim, dicts, trainingOptions)
  self.args = onmt.utils.ExtendedCmdLine.getModuleOpts(args, options)

  self.options = trainingOptions or args
  self.model = model
  self.optim = optim
  self.dicts = dicts
end

function Saver:_save(filePath, info)
  info.learningRate = self.optim:getLearningRate()
  info.optimStates = self.optim:getStates()
  info.rngStates = onmt.utils.Cuda.getRNGStates()

  local data = {
    models = {},
    options = self.options,
    info = info,
    dicts = self.dicts
  }

  for k, v in pairs(self.model.models) do
    if v.serialize then
      data.models[k] = v:serialize()
    else
      data.models[k] = v
    end
  end

  torch.save(filePath, data)
end

--[[ Save the model and the current epoch state. ]]
function Saver:saveIteration(iteration, epochState, batchOrder)
  local info = {}
  info.iteration = iteration + 1
  info.epoch = epochState.epoch
  info.batchOrder = batchOrder

  local filePath = string.format('%s_checkpoint.t7', self.args.save_model)

  _G.logger:info('Saving checkpoint to \'' .. filePath .. '\'...')

  -- Succeed serialization before overriding existing file
  self:_save(filePath .. '.tmp', info)
  os.rename(filePath .. '.tmp', filePath)
end

--[[ Save the model with validation score. ]]
function Saver:saveEpoch(validPpl, epochState)
  local info = {}
  info.validPpl = validPpl
  info.epoch = epochState.epoch + 1
  info.iteration = 1
  info.trainTimeInMinute = epochState:getTime() / 60

  local filePath = string.format('%s_epoch%d_%.2f.t7', self.args.save_model, epochState.epoch, validPpl)

  _G.logger:info('Saving checkpoint to \'' .. filePath .. '\'...')

  self:_save(filePath, info)
end

return Saver
