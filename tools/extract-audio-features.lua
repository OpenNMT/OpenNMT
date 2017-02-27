-- extraction feature tool for audio file

local audiolib require 'audio'

local function hsec(time)
  local s = '00000000'..tostring(math.floor(time*100))
  return s:sub(#s-6)
end

onmt = {
  utils = {
    ExtendedCmdLine = require 'onmt.utils.ExtendedCmdLine',
    String = require 'onmt.utils.String',
    Table = require 'onmt.utils.Table'
  }
}

require 'tools.utils.audiotool'

local cmd = onmt.utils.ExtendedCmdLine.new('extract-audio-features.lua')

-- Options declaration.
local options = {
  {'-id',         '',    [[Identifier for the voice file in the extracted features.]]},
  {'-audio',      '',    [[Path to an audio file.]],
                           {valid=onmt.utils.ExtendedCmdLine.nonEmpty}},
  {'-stm'  ,      '',    [[Path to ARPA stm file - if provided, features are extracted by sentence - name of each segment id
                           is ID-BEGIN-END - where BEGIN and END are hundredth of seconds.]]},
  {'-save_feats', '',    [[Path to file to save in torch format, if empty dump features in text format on stdout.]]}
}

cmd:setCmdLineOptions(options, 'General')
audiotool.declareOpts(cmd)

local opt = cmd:parse(arg)

local allfeats = {}

local saudio, samplerate = audio.load(opt.audio)

local audiotool = audiotool.new(opt)

local segments = {}

if opt.stm then
  local stm = io.open(opt.stm, "r")

  while 1 do
    line = stm:read()
    if not line then
      break
    end
    local fields = {}
    for f in line:gmatch'([^%s]+)' do
      table.insert(fields, f)
      if #fields == 7 then
        break
      end
    end
    if fields[7] ~= 'ignore_time_segment_in_scoring' then
      table.insert(segments, {id=fields[1]..'-'..hsec(fields[4])..'-'..hsec(fields[5]), range={fields[2],fields[4],fields[5]}})
    end
  end
else
  segments = {id=opt.id}
end

local feats = audiotool:extractFeats(saudio, samplerate)

for _,s in ipairs(segments) do
  local sbegin=1
  local send=#feats
  if s.range then
    sbegin=math.floor(s.range[1]/opt.winstep)
    send=math.ceil(s.range[1]/opt.winstep)
  end

  if opt.save_feats == '' then
    print(s.id..'  [')

    for i=1, feats:size(1) do
      local s = ''
      for j=1, feats:size(2) do
        s = s .. '  ' .. tostring(string.format('%0.6f',feats[{i,j}]))
      end
      if i == feats:size(1) then
        s = s .. ' ]'
      end
      print(s)
    end
  else
    table.insert(allfeats, {id=s.id, feats=feats})
    torch.save(opt.save_feats,allfeats)
  end
end
