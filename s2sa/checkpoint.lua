require 'torch'

local Checkpoint = torch.class("Checkpoint")

function Checkpoint:__init(args)
  self.options = args.options
  self.save_path = self.options.savefile
  self.save_every = self.options.save_every
  self.intermediate_save = self.options.intermediate_save

  self.optim = args.optim
  self.layers = args.layers

  self.model_info = {
    iteration = 0,
    epoch = 0,
    epochs = self.options.epochs,
    train_script_path = args.script_path,
    start_decay_at = self.optim.start_decay_at,
    lr_decay = self.optim.lr_decay
  }
end

function Checkpoint:save(file_path)
  self.model_info.learning_rate = self.optim:get_rate()

  print('saving checkpoint to ' .. file_path)
  torch.save(file_path, {self.layers, self.options, self.model_info})
end

function Checkpoint:save_iteration(iteration, bookkeeper)
  self.model_info.iteration = iteration

  if self.intermediate_save > 0 and iteration % self.intermediate_save == 0 then
    self.model_info.epoch = bookkeeper.epoch
    self.model_info.epoch_status = bookkeeper:get_status()

    local file_path = string.format('%s_checkpoint.t7', self.save_path)
    if iteration == 0 then
      self:save(file_path)
    else
      self:save(file_path .. '.tmp')
      os.rename(file_path .. '.tmp', file_path)
    end
  end
end

function Checkpoint:save_epoch(score, bookkeeper)
  if bookkeeper.epoch % self.save_every == 0 then
    self.model_info.score = score
    self.model_info.epoch = bookkeeper.epoch
    self.model_info.epoch_status = bookkeeper:get_status()
    self.model_info.train_time_in_minute = bookkeeper:get_time() / 60

    self:save(string.format('%s_epoch%d_%.2f.t7', self.save_path, self.model_info.epoch, score))
  end

  --reinit epoch data for next epoch save
  self.model_info.iteration = 0
end

function Checkpoint:save_final()
  self:save(string.format('%s_final.t7', self.save_path))
end


return Checkpoint
