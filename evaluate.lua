local beam = require 's2sa.beam'
local path = require 'pl.path'

local function main()
  beam.init(arg)
  local opt = beam.getOptions()

  assert(path.exists(opt.src_file), 'src_file does not exist')

  local file = io.open(opt.src_file, "r")
  local out_file = io.open(opt.output_file,'w')
  for line in file:lines() do
    local result, nbests = beam.search(line)
    out_file:write(result .. '\n')

    for n = 1, #nbests do
      out_file:write(nbests[n] .. '\n')
    end
  end

  out_file:close()
end

main()
