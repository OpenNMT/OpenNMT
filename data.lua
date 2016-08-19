--
-- Manages encoder/decoder data matrices.
--

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
  if opt.start_symbol == 0 then
    self.source_l:add(-2)
    self.source = self.source[{{},{2, self.source:size(2)-1}}]
  end
  self.batch_idx = f:read('batch_idx'):all()

  self.target_size = f:read('target_size'):all()[1]
  self.source_size = f:read('source_size'):all()[1]
  self.target_nonzeros = f:read('target_nonzeros'):all()

  if opt.use_chars_enc == 1 then
    self.source_char = f:read('source_char'):all()
    self.char_size = f:read('char_size'):all()[1]
    self.char_length = self.source_char:size(3)
    if opt.start_symbol == 0 then
      self.source_char = self.source_char[{{}, {2, self.source_char:size(2)-1}}]
    end
  end

  if opt.use_chars_dec == 1 then
    self.target_char = f:read('target_char'):all()
    self.char_size = f:read('char_size'):all()[1]
    self.char_length = self.target_char:size(3)
  end

  self.length = self.batch_l:size(1)
  self.seq_length = self.target:size(2)
  self.batches = {}
  local max_source_l = self.source_l:max()
  local source_l_rev = torch.ones(max_source_l):long()
  for i = 1, max_source_l do
    source_l_rev[i] = max_source_l - i + 1
  end
  for i = 1, self.length do
    local source_i, target_i
    local target_output_i = self.target_output:sub(self.batch_idx[i],self.batch_idx[i]
      +self.batch_l[i]-1, 1, self.target_l[i])
    local target_l_i = self.target_l_all:sub(self.batch_idx[i],
      self.batch_idx[i]+self.batch_l[i]-1)
    if opt.use_chars_enc == 1 then
      source_i = self.source_char:sub(self.batch_idx[i],
        self.batch_idx[i] + self.batch_l[i]-1, 1,
        self.source_l[i]):transpose(1,2):contiguous()
    else
      source_i = self.source:sub(self.batch_idx[i], self.batch_idx[i]+self.batch_l[i]-1,
        1, self.source_l[i]):transpose(1,2)
    end
    if opt.reverse_src == 1 then
      source_i = source_i:index(1, source_l_rev[{{max_source_l-self.source_l[i]+1,
            max_source_l}}])
    end

    if opt.use_chars_dec == 1 then
      target_i = self.target_char:sub(self.batch_idx[i],
        self.batch_idx[i] + self.batch_l[i]-1, 1,
        self.target_l[i]):transpose(1,2):contiguous()
    else
      target_i = self.target:sub(self.batch_idx[i], self.batch_idx[i]+self.batch_l[i]-1,
        1, self.target_l[i]):transpose(1,2)
    end
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
    if opt.gpuid >= 0 then --if multi-gpu, source lives in gpuid1, rest on gpuid2
      cutorch.setDevice(opt.gpuid)
      source_input = source_input:cuda()
      if opt.gpuid2 >= 0 then
        cutorch.setDevice(opt.gpuid2)
      end
      target_input = target_input:cuda()
      target_output = target_output:cuda()
      target_l_all = target_l_all:cuda()
    end
    return {target_input, target_output, nonzeros, source_input,
      batch_l, target_l, source_l, target_l_all}
  end
end

return data
