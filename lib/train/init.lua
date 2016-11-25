require('torch')

train = {}

train.Checkpoint = require('lib.train.checkpoint')
train.EpochState = require('lib.train.epoch_state')
train.Optim = require('lib.train.optim')

return train
