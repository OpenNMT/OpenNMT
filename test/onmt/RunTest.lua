local tester = ...

local runTest = torch.TestSuite()

function runTest.basic()
  local file = io.popen([[th preprocess.lua 2>&1]])
  local output = file:read('*all')
  local exit = file:close()
  tester:eq(exit,nil)
  tester:assertgt(string.find(output, "invalid argument"),0)
end

function runTest.run_real()
  local file = io.popen([[th preprocess.lua -train_src data/src-train.txt -train_tgt data/tgt-train.txt\
                          -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data tiny\
                          -src_vocab_size 50 -tgt_vocab_size 50 -src_seq_length 20 -tgt_seq_length 15\
                          2>&1]])
  local output = file:read('*all')
  local status = file:close()
  tester:assert(status ~= nil, "preprocess failed")
  if status then
    local v = tonumber(string.match(output, "Prepared (%d+) sentences"))
    tester:eq(v, 2984)

    file = io.popen([[th train.lua -data tiny-train.t7 -save_model tiny -end_epoch 2\
                      -rnn_size 10 -word_vec_size 10 -profiler 2>&1]])
    output = file:read('*all')
    status = file:close()
    tester:assert(status, "train failed")
    if status then
      local ppls={}
      for ppl in string.gmatch(output,"Validation perplexity: (%d+%.%d+)") do
        table.insert(ppls, tonumber(ppl))
        local filename="tiny_epoch"..#ppls.."_"..ppl..".t7"
        file=io.open(filename)
        tester:assert(file ~= nil, "cannot open file: "..filename)
        file:close()
        os.remove(filename)
      end

      tester:eq(#ppls,2)
      tester:eq(ppls,{9.95,7.19},1)
      tester:assert(string.find(output, "INFO] profile:") > 0, "cannot find profiling information")
    end
  else
    print(output)
  end
  os.remove("tiny-train.t7")
  os.remove("tiny.src.dict")
  os.remove("tiny.tgt.dict")
end

return runTest
