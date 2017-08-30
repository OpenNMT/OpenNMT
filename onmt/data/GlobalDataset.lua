--[[ Global Dataset class ]]

local GlobalDataset, _ = torch.class("GlobalDataset", "Dataset")
local paths = require 'paths'
local path = require 'pl.path'

local options = {
  {
    '-gd_train_dir', '',
    [[Path to training files directory.]],
    {
      valid = onmt.utils.ExtendedCmdLine.fileNullOrExists
    }
  },
  {
    '-gd_valid_dir', '',
    [[Path to valid files directory.]],
    {
      valid = onmt.utils.ExtendedCmdLine.fileNullOrExists,
      depends = function(opt) return opt.gd_train_dir == '' or opt.gd_train_dir ~= '' end
    }
  },
  {
    '-gd_sample', 0,
    [[If not null, extract sample from dataset.]]
  },
  {
    '-gd_features_vocabs_prefix', '',
    [[Path prefix to existing features vocabularies.]]
  },
  {
    '-gd_sample_dist', '',
    [[Configuration file with data class distribution to use for sampling training corpus.]],
    {
      valid = onmt.utils.ExtendedCmdLine.fileNullOrExists,
      depends = function(opt) return opt.gd_sample_dist == '' or opt.gd_sample > 0 end
    }
  }
}

local function prefix(data)
  if data.short then return data.short.."_" end
  return ""
end
local function nameWithSpace(data)
  if data.name then return " "..data.name end
  return ""
end

function GlobalDataset.declareOpts(cmd, modelClass)
  local data_type = modelClass.dataType()
  local datalist = onmt.data.Preprocessor.getDataList(data_type)
  for i = 1, #datalist do
    table.insert(options,
      {
        '-gd_'..prefix(datalist[i])..'vocab', '',
        "Path to an existing"..nameWithSpace(datalist[i]).." vocabulary.",
        {
          valid=onmt.utils.ExtendedCmdLine.fileNullOrExists
        }
      })
    table.insert(options,
      {
        '-gd_'..prefix(datalist[i])..'suffix', datalist[i].suffix,
        "Suffix for"..nameWithSpace(datalist[i]).." files in train/valid directories."
      })
    table.insert(options,
      {
        '-gd_'..prefix(datalist[i])..'seq_length', 50,
        "Maximum"..nameWithSpace(datalist[i])..[[ sequence length.]],
        {
          valid = onmt.utils.ExtendedCmdLine.isInt(1)
        }
      })
  end
  cmd:setCmdLineOptions(options, 'Global dataset')
end

function GlobalDataset:__init(args, modelClass)
  assert(args.gd_train_dir ~= '')
  _G.logger:info('Parsing training data from \''..args.gd_train_dir..'\':')
  local data_type = modelClass.dataType()
  local datalist = onmt.data.Preprocessor.getDataList(data_type)
  local firstSuffix = args["gd_"..prefix(datalist[1])..'suffix']
  self.totalCount = 0
  local totalError = 0
  self.files = {}
  for f in paths.iterfiles(args.gd_train_dir) do
    local flist = {}
    if f:sub(-firstSuffix:len()) == firstSuffix then
      local fprefix = f:sub(1, -firstSuffix:len()-1)
      table.insert(flist, paths.concat(args.gd_train_dir,f))
      local countLines = onmt.utils.FileReader.countLines(flist[1])
      local error = 0
      for i = 2, #datalist do
        local tfile = paths.concat(args.gd_train_dir,fprefix..args["gd_"..prefix(datalist[i])..'suffix'])
        table.insert(flist, tfile)
        if not path.exists(tfile) or onmt.utils.FileReader.countLines(tfile) ~= countLines then
          _G.logger:error('* invalid file - '..tfile..' - not aligned with '..f)
          error = error + 1
        end
      end
      if error == 0 then
        _G.logger:info('* Reading files \''..fprefix..'\' - '..countLines..' sentences')
        table.insert(self.files, {countLines, flist})
        self.totalCount = self.totalCount + countLines
      else
        totalError = totalError + 1
      end
    end
  end
  if totalError > 0 then
    _G.logger:error('Errors in training directory - fix them first')
    os.exit(0)
  end
  self.max_seq_length = {}
  for i = 1, #datalist do
    table.insert(self.max_seq_length, args["gd_"..prefix(datalist[i])..'seq_length'])
  end
  _G.logger:info(self.totalCount..' sentences in training directory')
  self.gd_sample = args.gd_sample
end

return GlobalDataset
