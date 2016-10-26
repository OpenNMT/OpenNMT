require 'torch'

local file_reader = require 's2sa.file_reader'

local parallel_file_reader = torch.class("parallel_file_reader")

function parallel_file_reader:__init(src_file, targ_file)
  self.src = file_reader.new(src_file)
  self.targ = file_reader.new(targ_file)
end

function parallel_file_reader:next()
  local src_line = self.src:next()
  local targ_line = self.targ:next()

  return src_line, targ_line
end

function parallel_file_reader:close()
  self.src:close()
  self.targ:close()
end

return parallel_file_reader
