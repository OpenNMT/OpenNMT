-- Class for saving and loading models during training.
local Checkpoint = torch.class('Checkpoint')

local options = {
  {'-train_from', '',  [[If training from a checkpoint then this is the path to the pretrained model.]],
                         {valid=onmt.utils.ExtendedCmdLine.fileNullOrExists}},
  {'-continue', false, [[If training from a checkpoint, whether to continue the training in the same configuration or not.]]}
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

-- structural parameters - the parameters with true can be changed dynamically in a train_from
local structural_parameters = {
  -- input feeding
  input_feed       = false,
  -- rnn parameters
  brnn_merge       = false,
  brnn             = false,
  dbrnn            = false,
  pdbrnn           = false,
  pdbrnn_reduction = false,
  rnn_size         = false,
  rnn_type         = false,
  -- dropout parameters
  dropout          = true,
  dropout_input    = false,
  -- layers of the NN
  layers           = false,
  -- feature parameters
  feat_merge       = false,
  feat_vec_exponent= false,
  feat_vec_size    = false,
  -- word embedding
  fix_word_vecs_enc= false,
  fix_word_vecs_dec= false,
  word_vec_size    = false,
  src_word_vec_size= false,
  tgt_word_vec_size= false,
  -- residual connections
  residual         = false
}

-- initialization parameters of the NN - can not be reused when restarting a run
local initialization_parameters = {
  model_type       = false,
  pre_word_vecs_dec= false,
  pre_word_vecs_enc= false,
  param_init       = false
}

function Checkpoint.loadFromCheckpoint(opt)
  local checkpoint = {}
  local param_changes = {}
  if opt.train_from:len() > 0 then
    _G.logger:info('Loading checkpoint \'' .. opt.train_from .. '\'...')

    checkpoint = torch.load(opt.train_from)
    local error

    for k,v in pairs(structural_parameters) do
      -- if parameter was set in commandline (and not default value)
      -- we need to check that we can actually change it
      if opt[k] and not opt[k..'_default'] and opt[k] ~= checkpoint.options[k] then
        if v == false then
          _G.logger:error('Cannot change dynamically parameters: %s', k)
          error = true
        else
          param_changes[k] = opt[k]
        end
      end
      opt[k] = checkpoint.options[k]
    end

    for k,_ in pairs(initialization_parameters) do
      if opt[k] and opt[k] ~= checkpoint.options[k] then
        _G.logger:error('Cannot change initialization parameters: %s', k)
        error = true
      end
    end

    if error then
      os.exit(1)
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
  return checkpoint, opt, param_changes
end

return Checkpoint
