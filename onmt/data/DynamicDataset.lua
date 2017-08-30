--[[ Dynamic Dataset class ]]

local DynamicDataset, _ = torch.class("DynamicDataset", "Dataset")

function DynamicDataset:__init(count, files,
                               dicts,
                               sample, sample_dist,
                               max_seq_length)
  self.count = count
  self.files = files
  self.dicts = dicts
  self.sample = sample
  self.sample_dist = sample_dist
  self.max_seq_length = max_seq_length
end



return DynamicDataset
