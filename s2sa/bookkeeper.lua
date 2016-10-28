require 'torch'

local Bookkeeper = torch.class("Bookkeeper")

function Bookkeeper:__init(args)
  self.learning_rate = args.learning_rate or 0
  self.data_size = args.data_size or 0
  self.epoch = args.epoch or 0

  self.timer = torch.Timer()

  self.train_nonzeros = 0
  self.train_loss = 0
  self.num_words_source = 0
  self.num_words_target = 0
end

function Bookkeeper:update(batch, loss)
  self.num_words_source = self.num_words_source + batch.size * batch.source_length
  self.num_words_target = self.num_words_target + batch.size * batch.target_length
  self.train_nonzeros = self.train_nonzeros + batch.target_non_zeros
  self.train_loss = self.train_loss + loss * batch.size
end

function Bookkeeper:log(batch_index)
  local time_taken = self.timer:time().real

  local stats = string.format('Epoch %d ; Batch %d/%d ; LR %.4f ; ',
                              self.epoch, batch_index, self.data_size, self.learning_rate)
  stats = stats .. string.format('Throughput %d/%d/%d total/src/targ tokens/sec ; ',
                                 (self.num_words_target + self.num_words_source) / time_taken,
                                 self.num_words_source / time_taken,
                                 self.num_words_target / time_taken)
  stats = stats .. string.format('PPL %.2f',
                                 math.exp(self.train_loss/self.train_nonzeros))

  print(stats)
end

function Bookkeeper:get_train_score()
  return math.exp(self.train_loss/self.train_nonzeros)
end

return Bookkeeper
