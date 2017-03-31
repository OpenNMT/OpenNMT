-- Class for saving and loading models during training.
local Checkpoint = torch.class('Checkpoint')

local options = {
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

function Checkpoint.declareOpts(cmd)
  cmd:setCmdLineOptions(options, 'Checkpoint')
end

function Checkpoint:__init(opt, model, optim, dicts)
  self.options = opt
  self.model = model
  self.optim = optim
  self.dicts = dicts

  self.savePath = self.options.save_model
end

function Checkpoint:save(filePath, info)
  info.learningRate = self.optim:getLearningRate()
  info.optimStates = self.optim:getStates()

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

--[[ Save the model and data in the middle of an epoch sorting the iteration. ]]
function Checkpoint:saveIteration(iteration, epochState, batchOrder, verbose)
  local info = {}
  info.iteration = iteration + 1
  info.epoch = epochState.epoch
  info.batchOrder = batchOrder

  local filePath = string.format('%s_checkpoint.t7', self.savePath)

  if verbose then
    _G.logger:info('Saving checkpoint to \'' .. filePath .. '\'...')
  end

  -- Succeed serialization before overriding existing file
  self:save(filePath .. '.tmp', info)
  os.rename(filePath .. '.tmp', filePath)
end

function Checkpoint:saveEpoch(validPpl, epochState, verbose)
  local info = {}
  info.validPpl = validPpl
  info.epoch = epochState.epoch + 1
  info.iteration = 1
  info.trainTimeInMinute = epochState:getTime() / 60

  local filePath = string.format('%s_epoch%d_%.2f.t7', self.savePath, epochState.epoch, validPpl)

  if verbose then
    _G.logger:info('Saving checkpoint to \'' .. filePath .. '\'...')
  end

  self:save(filePath, info)
end

function Checkpoint.loadFromCheckpoint(opt)
  local checkpoint = {}
  local paramChanges = {}
  if opt.train_from:len() > 0 then
    _G.logger:info('Loading checkpoint \'' .. opt.train_from .. '\'...')

    checkpoint = torch.load(opt.train_from)

    -- Reload and check options.
    for k,v in pairs(opt) do
      if k:sub(1, 1) ~= '_' then
        -- If an option was set by the user, check that we can actually change it.
        local isDefault = opt._is_default and opt._is_default[k]

        if opt._structural[k] then
          if not isDefault and v ~= checkpoint.options[k] then
            if opt._structural[k] == 0 then
              _G.logger:warning('Cannot change dynamically option -%s. Ignoring.', k)
            else
              paramChanges[k] = v
            end
          end
          opt[k] = checkpoint.options[k]
        end

        if opt._init_only[k] == true and not isDefault then
          _G.logger:warning('Cannot change initialization option -%s. Ignoring.', k)
        end
      end
    end

    -- Resume training from checkpoint
    if opt.continue then
      opt.optim = checkpoint.options.optim
      opt.learning_rate_decay = checkpoint.options.learning_rate_decay
      opt.start_decay_at = checkpoint.options.start_decay_at
      opt.curriculum = checkpoint.options.curriculum

      opt.decay = checkpoint.options.decay or opt.decay
      opt.min_learning_rate = checkpoint.options.min_learning_rate or opt.min_learning_rate
      opt.start_decay_ppl_delta = checkpoint.options.start_decay_ppl_delta or opt.start_decay_ppl_delta

      opt.learning_rate = checkpoint.info.learningRate
      opt.start_epoch = checkpoint.info.epoch
      opt.start_iteration = checkpoint.info.iteration

      _G.logger:info('Resuming training from epoch ' .. opt.start_epoch
                         .. ' at iteration ' .. opt.start_iteration .. '...')
    else
      checkpoint.info = nil
    end
  end

  return checkpoint, opt, paramChanges
end

return Checkpoint
