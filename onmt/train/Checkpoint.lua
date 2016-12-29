-- Class for saving and loading models during training.
local path = require('pl.path')

local Checkpoint = torch.class("Checkpoint")

function Checkpoint:__init(options, model, optim, dataset)
  self.options = options
  self.model = model
  self.optim = optim
  self.dataset = dataset

  self.savePath = self.options.save_model
end

function Checkpoint.declareOpts(cmd)
  cmd:text("")
  cmd:text("**Checkpoint options**")
  cmd:text("")

  cmd:option('-train_from', '', [[If training from a checkpoint then this is the path to the pretrained model.]])
  cmd:option('-continue', false, [[If training from a checkpoint, whether to continue the training in the same configuration or not.]])
end

function Checkpoint.init(opt)
  local checkpoint = {}
  if opt.train_from:len() > 0 then
    assert(path.exists(opt.train_from), 'checkpoint path invalid')

    if not opt.json_log then
      print('Loading checkpoint \'' .. opt.train_from .. '\'...')
    end

    checkpoint = torch.load(opt.train_from)

    -- TODO: currently this is much to MT specific.
    -- Checkpoint code should not know about these options.
    opt.layers = checkpoint.options.layers
    opt.rnn_size = checkpoint.options.rnn_size
    opt.brnn = checkpoint.options.brnn
    opt.brnn_merge = checkpoint.options.brnn_merge
    opt.input_feed = checkpoint.options.input_feed

    -- Resume training from checkpoint
    if opt.train_from:len() > 0 and opt.continue then
      opt.optim = checkpoint.options.optim
      opt.learning_rate_decay = checkpoint.options.learning_rate_decay
      opt.start_decay_at = checkpoint.options.start_decay_at
      opt.epochs = checkpoint.options.epochs
      opt.curriculum = checkpoint.options.curriculum

      opt.learning_rate = checkpoint.info.learningRate
      opt.optim_states = checkpoint.info.optimStates
      opt.start_epoch = checkpoint.info.epoch
      opt.start_iteration = checkpoint.info.iteration

      if not opt.json_log then
        print('Resuming training from epoch ' .. opt.start_epoch
                .. ' at iteration ' .. opt.start_iteration .. '...')
      end
    end
  end
  return checkpoint
end

function Checkpoint:save(filePath, info)
  info.learningRate = self.optim:getLearningRate()
  info.optimStates = self.optim:getStates()

  local data = {
    models = {},
    options = self.options,
    info = info,
    dicts = self.dataset.dicts
  }

  for k, v in pairs(self.model) do
    data.models[k] = v:serialize()
  end

  torch.save(filePath, data)
end

--[[ Save the model and data in the middle of an epoch sorting the iteration. ]]
function Checkpoint:saveIteration(iteration, epochState, batchOrder, verbose)
  local info = {}
  info.iteration = iteration + 1
  info.epoch = epochState.epoch
  info.epochStatus = epochState:getStatus()
  info.batchOrder = batchOrder

  local filePath = string.format('%s_checkpoint.t7', self.savePath)

  if verbose then
    print('Saving checkpoint to \'' .. filePath .. '\'...')
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
    print('Saving checkpoint to \'' .. filePath .. '\'...')
  end

  self:save(filePath, info)
end

return Checkpoint
