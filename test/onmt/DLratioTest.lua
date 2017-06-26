require('onmt.init')

local tester = ...

local DLratioTest = torch.TestSuite()

local ref = {
[[After 60 days of hard work , the scientists from Xi ' an Satellite Testing and Monitoring Center overcame various technical difficulties and successfully solved the malfunction of the Beidou Navigation Experimental Satellites ( BDNES ) .]],
[[Currently , the satellites are running smoothly and the devices on the satellites are functioning properly .]],
[[According to the person - in - charge , on the 3 rd Febuary 2007 , after the Beidou Navigation Experimental Satellite was launched at the Xichang Satellite Launch Center , the satellite could not function properly because the solar sailboard malfunctioned during its spread .]]}

local cand = {
[[After 60 days of fierce battle , the technical personnel of Xi ' an Satellite Monitoring and Control Center fought out several technical difficulties and successfully excluded the satellite satellite satellite malfunction .]],
[[At present , the satellite has good attitude and the instrument works normally .]],
[[On February 3 , 2007 , the Beidou Navigation Satellite launched a satellite in Xichang Satellite Launch Center , causing a malfunction of the satellite , according to officials concerned .]]}


local function tok(t)
  local tt = {}
  for i = 1, #t do
    local toks = {}
    for word in t[i]:gmatch'([^%s]+)' do
      table.insert(toks, word)
    end
    table.insert(tt, toks)
  end
  return tt
end

function DLratioTest.basic()
    local refs = tok(ref)
    local cands = tok(cand)
    local dlratio = onmt.scorers['dlratio'](cands, refs)
    tester:eq(dlratio, 0.54, 0.01)
end

return DLratioTest
