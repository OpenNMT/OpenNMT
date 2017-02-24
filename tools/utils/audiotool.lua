local audiotool = torch.class('audiotool')
local signal = require 'signal'
local audio = require 'audio'

-- MFSC and MFCC calculation inspired from
-- https://github.com/jameslyons/python_speech_features
-- http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/

-- Options declaration.
local options = {
  { '-feats',   'mfcc', [[Features to extract.]], { enum={'mfcc','mfsc'}} },
  { '-winlen',   0.025, [[The length of the analysis window in seconds.]] },
  { '-winstep',  0.010, [[the step between successive windows in seconds.]] },
  { '-wintype', 'rect', [[Windows type.]],
                  { enum={'rect', 'hamming', 'hann', 'bartlett'}} },
  { '-numcep',      13, [[The number of cepstrum to return.]] },
  { '-nfilt',       26, [[The number of filters in the filterbank.]] },
  { '-lowfreq',      0, [[Lowest band edge of mel filters in Herz.]] },
  { '-highfreq',    -1, [[Highest band edge of mel filters in Herz, if -1 default to samplerate/2.]] },
  { '-preemph',   0.97, [[Apply preemphasis filter with preemph as coefficient. 0 is no filter.]] },
  { '-ceplifter',   22, [[Apply a lifter to final cepstral coefficients. 0 is no lifter. ]] },
  { '-appendEnergy', 1, [[If non zero, the first cepstral coefficient is replaced
                              with the log of the total frame energy.]] },
  { '-cmvn',         1, [[If non zero, apply cepstral mean and variance normalization.]],
                  { valid=onmt.utils.ExtendedCmdLine.isInt(0,1)} },
  { '-delta_step',   5, [[Value of N in delta calculation for mfcc, 0 to disable delta calculation.]],
                  { valid=onmt.utils.ExtendedCmdLine.isInt(0)} }
}

function audiotool.declareOpts(cmd)
  cmd:setCmdLineOptions(options, 'Audio')
end

function audiotool.getModuleOpts(args)
  return onmt.utils.ExtendedCmdLine.getModuleOpts(args, options)
end

function audiotool:__init(args)
  self.args = onmt.utils.ExtendedCmdLine.getModuleOpts(args, options)
end

-- herz to mel conversion
local function herz2mel(f)
  if torch.isTensor(f) then
    return 1127*torch.log(1+f/700)
  else
    return 1127*math.log(1+f/700)
  end
end

-- mel to herz conversion
local function mel2herz(m)
  if torch.isTensor(m) then
    return 700*(torch.exp(m/1127)-1)
  else
    return 700*(math.exp(m/1127)-1)
  end
end

--[[Perform preemphasis on the input signal.

  * saudio: The signal to filter.
  * coeff: The preemphasis coefficient. 0 is no filter.
  * returns the filtered signal.
]]
local function preemphasis(saudio, coeff)
  local ret = saudio:clone()
  ret:narrow(1, 2, ret:size(1)-1):add(-coeff*ret:narrow(1, 1, ret:size(1)-1))
  return ret
end

--[[Build Mel-filterbank

  * n: number of filters
  * lowfreq: lowest frequency
  * highfreq: highest frequency
  * samplerate: sample rate
  * winsize: the windows size
]]
local function get_filterbanks(n, winsize, lowfreq, highfreq, samplerate)
  -- compute points evenly spaced in mels
  local lowmel = herz2mel(lowfreq)
  local highmel = herz2mel(highfreq)
  local melpoints = torch.linspace(lowmel, highmel, n+2)
  -- our points are in Hz, but we use fft bins, so we have to convert
  -- from Hz to fft bin number
  local bin = torch.floor((winsize+1)*mel2herz(melpoints)/samplerate)
  local fbank = torch.zeros(n,math.floor(winsize)/2+1)

  for j=0,n-1 do
    for i=math.floor(bin[j+1]), math.floor(bin[j+2])-1 do
      fbank[{j+1,i+1}] = (i - bin[j+1]) / (bin[j+2]-bin[j+1])
    end
    for i=math.floor(bin[j+2]), math.floor(bin[j+3])-1 do
      fbank[{j+1,i+1}] = (bin[j+3]-i) / (bin[j+3]-bin[j+2])
    end
  end
  return fbank
end

--[[Compute delta features from a feature vector sequence.

  * feat: a tensor of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature vector.
  * N: For each frame, calculate delta features based on preceding and following N frames
  * returns a tensor containing delta features. Each row holds 1 delta feature vector.
]]
local function delta(feat, N)
  local NUMFRAMES = feat:size(1)
  local denominator = 0
  for i=1, N do
    denominator = denominator + 2*i*i
  end
  local delta_feat = torch.zeros(feat:size(1), feat:size(2))
  local padded = torch.Tensor(feat:size(1)+2*N, feat:size(2))
  for i=1,N do
    padded:narrow(1, i, 1):copy(feat:narrow(1,1,1))
    padded:narrow(1, N+i+feat:size(2), 1):copy(feat:narrow(1, feat:size(1), 1))
  end
  for t=1, NUMFRAMES do
    delta_feat[t] = (torch.linspace(-N, N+1, 2*N+1):resize(1,2*N+1)*padded:narrow(1, t, 2*N+1)) / denominator
  end
  return delta_feat
end

--[[Apply a cepstral lifter to the matrix of cepstra. T
    This has the effect of increasing the
    magnitude of the high frequency DCT coeffs.

    * cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
    * L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
]]
local function lifter(cepstra, L)
  if L > 0 then
    local ncoeff = cepstra:size(2)
    local n = torch.linspace(0, ncoeff-1, ncoeff)
    local lift = 1 + (L/2)*torch.sin(math.pi*n/L)
    return cepstra:cmul(lift:repeatTensor(cepstra:size(1)))
  end
  return cepstra
end

--[[Compute MFCC features from an audio signal.

  * saudio: the audio signal from which to compute features. Should be a 1D or 2D: N*1 tensor
  * samplerate: the samplerate of the signal we are working with.
  * returns a tensor of size N*numcep
]]
function audiotool:mfcc(saudio, samplerate)
  local energy, feats = self:mfsc(saudio, samplerate)

  feats = signal.dct2(torch.log(feats)):narrow(2,1,self.args.numcep)
  -- TODO normalize DCT (ortho)

  feats = lifter(feats, self.args.ceplifter)

  if self.args.appendEnergy ~= 0 then
    feats:select(2,1):copy(torch.log(energy))
  end

  if self.args.cmvn ~= 0 then
    local mean = feats:mean()
    local std = feats:std()
    feats = (feats - mean)/std
  end

  if self.args.delta_step > 0 then
    local deltaf = delta(feats, self.args.delta_step)
    local delta2f = delta(deltaf, self.args.delta_step)
    feats = torch.cat({feats, deltaf, delta2f})
  end
  return feats
end

--[[Compute MFSC features from an audio signal.

  * saudio: the audio signal from which to compute features. Should be a 1D or 2D: N*1 tensor
  * samplerate: the samplerate of the signal we are working with.
  * returns energy, a tensor of size N*nfilt
]]
function audiotool:mfsc(saudio, samplerate)
  assert(saudio:dim()==1 or (saudio:dim()==2 and saudio:size(2)==1))
  if self.args.highfreq == -1 then
    self.args.highfreq = samplerate/2
  end
  saudio:resize(saudio:size(1))
  -- normalizing value / 65536 and preemphasis
  saudio = preemphasis(saudio/65536, self.args.preemph)

  local winsize = math.floor(self.args.winlen*samplerate)
  local stride = math.floor(self.args.winstep*samplerate)

  local stft = audio.stft(saudio, winsize, self.args.wintype, stride)
  local pspec = torch.Tensor(stft:size(1), stft:size(2)):zero()
  local energy = torch.Tensor(stft:size(1))
  for n=1,stft:size(1) do
    pspec:select(1,n):add(torch.cmul(stft[{n,{},1}],stft[{n,{},1}])/winsize)
    pspec:select(1,n):add(torch.cmul(stft[{n,{},2}],stft[{n,{},2}])/winsize)
    -- make sure energy is not zero
    energy[n]=pspec:select(1,n):sum()+0.0000000001
  end

  local fb = get_filterbanks(self.args.nfilt, winsize, self.args.lowfreq, self.args.highfreq, samplerate)

  local feat = pspec*fb:t()

  return energy, feat
end

function audiotool:extractFeats(saudio, samplerate)
  if self.args.feats == 'mfcc' then
    return self:mfcc(saudio, samplerate)
  else
    return self:mfsc(saudio, samplerate)
  end
end

return audiotool
