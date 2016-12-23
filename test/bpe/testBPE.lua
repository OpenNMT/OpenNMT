local BPE = require ('tools.utils.BPE')

--getPairs('abcdimprovement联合国')
local bpe = BPE.new('/home/deng/projects/systran/OpenNMT/test/bpe/testcode.bpe')
print (bpe.codes)
print (bpe:encode('abcdimprovement联合国'))
print (bpe:encode('a'))
print (bpe:encode('国'))
