local train = {}

train.Trainer = require('onmt.train.Trainer')
train.Checkpoint = require('onmt.train.Checkpoint')
train.EpochState = require('onmt.train.EpochState')
train.Optim = require('onmt.train.Optim')

return train
