require 'torch'

local EpochState = torch.class("EpochState")

function EpochState:__init(epoch, status)
  self.epoch = epoch

  if status ~= nil then
    self.status = status
  else
    self.status = {}
    self.status.train_nonzeros = 0
    self.status.train_loss = 0
  end

  self.timer = torch.Timer()
  self.num_words_source = 0
  self.num_words_target = 0

  print('')
end

function EpochState:update(batch, loss)
  self.num_words_source = self.num_words_source + batch.size * batch.source_length
  self.num_words_target = self.num_words_target + batch.size * batch.target_length

  self.status.train_nonzeros = self.status.train_nonzeros + batch.target_non_zeros
  self.status.train_loss = self.status.train_loss + loss * batch.size
end

function EpochState:log(batch_index, data_size, learning_rate)
  local time_taken = self:get_time()

  local stats = ''

  stats = stats .. string.format('Epoch %d ; Batch %d/%d ; LR %.4f ; ',
                                 self.epoch, batch_index, data_size, learning_rate)
  stats = stats .. string.format('Throughput %d/%d/%d total/src/targ tokens/sec ; ',
                                 (self.num_words_target + self.num_words_source) / time_taken,
                                 self.num_words_source / time_taken,
                                 self.num_words_target / time_taken)
  stats = stats .. string.format('PPL %.2f', self:get_train_ppl())

  print(stats)
end

function EpochState:get_train_ppl()
  return math.exp(self.status.train_loss / self.status.train_nonzeros)
end

function EpochState:get_time()
  return self.timer:time().real
end

function EpochState:get_status()
  return self.status
end

return EpochState
