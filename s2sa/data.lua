--
-- Manages encoder/decoder data matrices.
--

require 'hdf5'

local data = torch.class("data")

function data:__init(opt, data_file)
  local f = hdf5.open(data_file, 'r')

  self.source = f:read('source'):all()
  self.target = f:read('target'):all()
  self.target_output = f:read('target_output'):all()
  self.target_l = f:read('target_l'):all() --max target length each batch
  self.target_l_all = f:read('target_l_all'):all()
  self.target_l_all:add(-1)
  self.batch_l = f:read('batch_l'):all()
  self.source_l = f:read('batch_w'):all() --max source length each batch

  self.batch_idx = f:read('batch_idx'):all()

  self.target_size = f:read('target_size'):all()[1]
  self.source_size = f:read('source_size'):all()[1]
  self.target_nonzeros = f:read('target_nonzeros'):all()

  self.length = self.batch_l:size(1)
  self.seq_length = self.target:size(2)
  self.batches = {}

  for i = 1, self.length do
    local source_i = self.source:sub(self.batch_idx[i], self.batch_idx[i]+self.batch_l[i]-1,
        1, self.source_l[i]):transpose(1,2)
    local target_i = self.target:sub(self.batch_idx[i], self.batch_idx[i]+self.batch_l[i]-1,
        1, self.target_l[i]):transpose(1,2)
    local target_output_i = self.target_output:sub(self.batch_idx[i],self.batch_idx[i]
      +self.batch_l[i]-1, 1, self.target_l[i])
    local target_l_i = self.target_l_all:sub(self.batch_idx[i],
      self.batch_idx[i]+self.batch_l[i]-1)

    table.insert(self.batches, {target_i,
        target_output_i:transpose(1,2),
        self.target_nonzeros[i],
        source_i,
        self.batch_l[i],
        self.target_l[i],
        self.source_l[i],
        target_l_i})
  end
end

function data:size()
  return self.length
end

function data.__index(self, idx)
  if type(idx) == "string" then
    return data[idx]
  else
    local target_input = self.batches[idx][1]
    local target_output = self.batches[idx][2]
    local nonzeros = self.batches[idx][3]
    local source_input = self.batches[idx][4]
    local batch_l = self.batches[idx][5]
    local target_l = self.batches[idx][6]
    local source_l = self.batches[idx][7]
    local target_l_all = self.batches[idx][8]

    if opt.gpuid > 0 then --if multi-gpu, source lives in gpuid1, rest on gpuid2
      source_input = source_input:cuda()
      target_input = target_input:cuda()
      target_output = target_output:cuda()
      target_l_all = target_l_all:cuda()
    end
    return {target_input, target_output, nonzeros, source_input,
            batch_l, target_l, source_l, target_l_all}
  end
end

return data
