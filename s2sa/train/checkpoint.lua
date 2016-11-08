require 'torch'

local Checkpoint = torch.class("Checkpoint")

function Checkpoint:__init(args)
  self.options = args.options
  self.save_path = self.options.savefile
  self.model = args.model
  self.optim = args.optim
end

function Checkpoint:save(file_path, info)
  info.learning_rate = self.optim:get_learning_rate()
  info.optim_states = self.optim:get_states()

  local data = {
    model = self.model,
    options = self.options,
    info = info
  }

  torch.save(file_path, data)
end

function Checkpoint:save_iteration(iteration, epoch_state, batch_order)
  local info = {}
  info.iteration = iteration + 1
  info.epoch = epoch_state.epoch
  info.epoch_status = epoch_state:get_status()
  info.batch_order = batch_order

  local file_path = string.format('%s_checkpoint.t7', self.save_path)

  print('Saving checkpoint to ' .. file_path .. '...')

  -- succeed serialization before overriding existing file
  self:save(file_path .. '.tmp', info)
  os.rename(file_path .. '.tmp', file_path)
end

function Checkpoint:save_epoch(valid_ppl, epoch_state)
  local info = {}
  info.valid_ppl = valid_ppl
  info.epoch = epoch_state.epoch + 1
  info.iteration = 1
  info.train_time_in_minute = epoch_state:get_time() / 60

  local file_path = string.format('%s_epoch%d_%.2f.t7', self.save_path, epoch_state.epoch, valid_ppl)

  print('Saving checkpoint to ' .. file_path .. '...')
  self:save(file_path, info)
end

return Checkpoint
