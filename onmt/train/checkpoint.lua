-- Class for saving and loading models during training.
local Checkpoint = torch.class("Checkpoint")

function Checkpoint:__init(options, model, optim, dataset)
  self.options = options
  self.model = model
  self.optim = optim
  self.dataset = dataset

  self.save_path = self.options.save_file
end

function Checkpoint:save(file_path, info)
  info.learning_rate = self.optim:get_learning_rate()
  info.optim_states = self.optim:get_states()

  local data = {
    models = {},
    options = self.options,
    info = info,
    dicts = self.dataset.dicts
  }

  for k, v in pairs(self.model) do
    data.models[k] = v:serialize()
  end

  torch.save(file_path, data)
end

--[[ Save the model and data in the middle of an epoch sorting the iteration. ]]
function Checkpoint:save_iteration(iteration, epoch_state, batch_order, verbose)
  local info = {}
  info.iteration = iteration + 1
  info.epoch = epoch_state.epoch
  info.epoch_status = epoch_state:get_status()
  info.batch_order = batch_order

  local file_path = string.format('%s_checkpoint.t7', self.save_path)

  if verbose then
    print('Saving checkpoint to \'' .. file_path .. '\'...')
  end

  -- Succeed serialization before overriding existing file
  self:save(file_path .. '.tmp', info)
  os.rename(file_path .. '.tmp', file_path)
end

function Checkpoint:save_epoch(valid_ppl, epoch_state, verbose)
  local info = {}
  info.valid_ppl = valid_ppl
  info.epoch = epoch_state.epoch + 1
  info.iteration = 1
  info.train_time_in_minute = epoch_state:get_time() / 60

  local file_path = string.format('%s_epoch%d_%.2f.t7', self.save_path, epoch_state.epoch, valid_ppl)

  if verbose then
    print('Saving checkpoint to \'' .. file_path .. '\'...')
  end

  self:save(file_path, info)
end

return Checkpoint
