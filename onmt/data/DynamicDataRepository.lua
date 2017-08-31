--[[ Global Dataset class ]]

local DynamicDataRepository, _ = torch.class("DynamicDataRepository")
local paths = require 'paths'
local path = require 'pl.path'

local options = {
  {
    '-train_dir', '',
    [[Path to training files directory.]],
    {
      valid = onmt.utils.ExtendedCmdLine.fileNullOrExists
    }
  },
  {
    '-valid_dir', '',
    [[Path to valid files directory.]],
    {
      valid = onmt.utils.ExtendedCmdLine.fileNullOrExists,
      depends = function(opt) return opt.train_dir == '' or opt.train_dir ~= '' end
    }
  },
  {
    '-features_vocabs_prefix', '',
    [[Path prefix to existing features vocabularies.]]
  },
  {
    '-sample_dist', '',
    [[Configuration file with data class distribution to use for sampling training corpus.]],
    {
      valid = onmt.utils.ExtendedCmdLine.fileNullOrExists,
      depends = function(opt) return opt.sample_dist == '' or opt.sample > 0 end
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

function DynamicDataRepository.declareOpts(cmd, modelClass)
  local data_type = modelClass.dataType()
  local datalist = onmt.data.Preprocessor.getDataList(data_type)
  for i = 1, #datalist do
    table.insert(options,
      {
        '-'..prefix(datalist[i])..'vocab', '',
        "Path to an existing"..nameWithSpace(datalist[i]).." vocabulary.",
        {
          valid=onmt.utils.ExtendedCmdLine.fileNullOrExists
        }
      })
    table.insert(options,
      {
        '-'..prefix(datalist[i])..'suffix', datalist[i].suffix,
        "Suffix for"..nameWithSpace(datalist[i]).." files in train/valid directories."
      })
    table.insert(options,
      {
        '-'..prefix(datalist[i])..'seq_length', 50,
        "Maximum"..nameWithSpace(datalist[i])..[[ sequence length.]],
        {
          valid = onmt.utils.ExtendedCmdLine.isInt(1)
        }
      })
  end
  cmd:setCmdLineOptions(options, 'Global dataset')
end

local function parseDirectory(args, datalist, type)
  local dir = args[type.."_dir"]
  assert(dir ~= '', 'missing \''..type..'_dir\' parameter')
  _G.logger:info('Parsing '..type..' data from \''..dir..'\':')
  local firstSuffix = args[prefix(datalist[1])..'suffix']
  local totalCount = 0
  local totalError = 0
  local list_files = {}
  for f in paths.iterfiles(dir) do
    local flist = {}
    if f:sub(-firstSuffix:len()) == firstSuffix then
      local fprefix = f:sub(1, -firstSuffix:len()-1)
      table.insert(flist, paths.concat(dir,f))
      local countLines = onmt.utils.FileReader.countLines(flist[1])
      local error = 0
      for i = 2, #datalist do
        local tfile = paths.concat(dir,fprefix..args[prefix(datalist[i])..'suffix'])
        table.insert(flist, tfile)
        if not path.exists(tfile) or onmt.utils.FileReader.countLines(tfile) ~= countLines then
          _G.logger:error('* invalid file - '..tfile..' - not aligned with '..f)
          error = error + 1
        end
      end
      if error == 0 then
        _G.logger:info('* Reading files \''..fprefix..'\' - '..countLines..' sentences')
        table.insert(list_files, {countLines, flist})
        totalCount = totalCount + countLines
      else
        totalError = totalError + 1
      end
    end
  end
  if totalError > 0 then
    _G.logger:error('Errors in training directory - fix them first')
    os.exit(0)
  end
  if totalCount == 0 then
    _G.logger:error('No '..type..' data found in directory \''..dir..'\'')
    os.exit(0)
  end
  _G.logger:info(totalCount..' sentences in '..type..' directory')
  return totalCount, list_files
end

function DynamicDataRepository:getTrain()
  return onmt.data.DynamicDataset.new(self.train_total, self.train_files,
                                      self.dicts,
                                      self.sample, self.sample_dist,
                                      self.max_seq_length)
end

function DynamicDataRepository:getValid()
  local dataset = onmt.data.DynamicDataset.new(self.valid_total, self.valid_files,
                                               self.dicts,
                                               self.sample, self.sample_dist,
                                               self.max_seq_length)
  return onmt.data.Dataset.new(dataset.getData())
end

function DynamicDataRepository:__init(args, modelClass)
  self.dataType = modelClass.dataType()
  local datalist = onmt.data.Preprocessor.getDataList(self.dataType)
  self.dicts = {}

  self.train_total, self.train_files = parseDirectory(args, datalist, 'train')
  self.valid_total, self.valid_files = parseDirectory(args, datalist, 'valid')

  -- check vocabularies and build seq_length
  self.seq_length = {}
  for i = 1, #datalist do
    table.insert(self.seq_length, args[prefix(datalist[i])..'seq_length'])
    local vocab_file = args[prefix(datalist[i])..'vocab']
    if vocab_file == '' then
      _G.logger:error("missing parameter "..prefix(datalist[i])..'vocab')
      os.exit(0)
    elseif not path.exists(vocab_file) then
      _G.logger:error("missing file "..vocab_file)
      os.exit(0)
    end
    self.dicts[datalist[i].short] = onmt.data.Vocabulary.init(datalist[i].name,
                                     self.train_files[1][2][i],
                                     vocab_file,
                                     0,
                                     0,
                                     args["features_vocabs_prefix"],
                                     function(s) return onmt.data.Preprocessor.isValid(s, self.seq_length[#self.seq_length]) end,
                                     false,
                                     false)
  end

  self.sample_train = args.sample

  self.preprocessor = onmt.data.Preprocessor.new(args, self.dataType)
  print(self.preprocessor:makeData('train', self.dicts))
end

return DynamicDataRepository
