require('onmt.init')

local tester = ...

local bleuTest = torch.TestSuite()

local ref1 = {
[[After 60 days of hard work , the scientists from Xi ' an Satellite Testing and Monitoring Center overcame various technical difficulties and successfully solved the malfunction of the Beidou Navigation Experimental Satellites ( BDNES ) .]],
[[Currently , the satellites are running smoothly and the devices on the satellites are functioning properly .]],
[[According to the person - in - charge , on the 3 rd Febuary 2007 , after the Beidou Navigation Experimental Satellite was launched at the Xichang Satellite Launch Center , the satellite could not function properly because the solar sailboard malfunctioned during its spread .]]}

local ref2 = {
[[After 60 days of trying to resolve technical problems , the technicians at the Xi ' an Satelllite Control Centre of China have successfully rectified the errors in the test flights of the Beidou satellite .]],
[[Currently , the satellite and its equipment are functioning as per normal .]],
[[According to the relative directors of the centre , the test launch of the Beidou satellite on 3 February , 2007 failed when the satellite ' s solar panels encountered problems while unfolding .]]}

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

function bleuTest.basic()
  local refs = { tok(ref1) }
  local candtok = tok(cand)
  -- one reference
  local bleu = onmt.scorers['bleu'](candtok, refs)
  tester:eq(bleu,0.15,0.01)
  table.insert(refs, tok(ref2))
  -- two references
  local details
  bleu, details = onmt.scorers['bleu'](candtok, refs)
  tester:eq(bleu,0.20,0.01)
  tester:assert(details:find("20.5") ~= 0)
  -- two references with order 5
  bleu = onmt.scorers['bleu'](candtok, refs, 5)
  tester:eq(bleu,0.12,0.01)
end

return bleuTest
