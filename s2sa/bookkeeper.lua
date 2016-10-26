require 'torch'

local Bookkeeper = torch.class("Bookkeeper")

function Bookkeeper:__init(args)
  self.print_frequency = args.print_frequency or 0
  self.learning_rate = args.learning_rate or 0
  self.data_size = args.data_size or 0
  self.epoch = args.epoch or 0

  self.timer = torch.Timer()
  self.start_time = self.timer:time().real
  self.train_nonzeros = 0
  self.train_loss = 0
  self.num_words_source = 0
  self.num_words_target = 0
end

function Bookkeeper:update(info)
  self.num_words_source = self.num_words_source + info.batch_size * info.source_size
  self.num_words_target = self.num_words_target + info.batch_size *info.target_size
  self.train_nonzeros = self.train_nonzeros + info.nonzeros
  self.train_loss = self.train_loss + info.loss * info.batch_size

  if info.batch_index % self.print_frequency == 0 then
    local time_taken = self.timer:time().real - self.start_time

    local stats = string.format('Epoch: %d, Batch: %d/%d, Batch size: %d, LR: %.4f, ',
      self.epoch, info.batch_index, self.data_size, info.batch_size, self.learning_rate)
    stats = stats .. string.format('PPL: %.2f, |Param|: %.2f, |GParam|: %.2f, ',
      math.exp(self.train_loss/self.train_nonzeros), info.param_norm, info.grad_norm)
    stats = stats .. string.format('Training: %d/%d/%d total/source/target tokens/sec',
      (self.num_words_target+self.num_words_source) / time_taken,
      self.num_words_source / time_taken,
      self.num_words_target / time_taken)
    print(stats)
  end
end

function Bookkeeper:get_train_score()
  return math.exp(self.train_loss/self.train_nonzeros)
end

return Bookkeeper
