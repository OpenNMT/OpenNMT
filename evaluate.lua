local beam = require 's2sa.beam'

function main()
  beam.init(arg)
  local opt = beam.getOptions()

  assert(path.exists(opt.src_file), 'src_file does not exist')

  local file = io.open(opt.src_file, "r")
  local out_file = io.open(opt.output_file,'w')
  for line in file:lines() do
    result, nbests = beam.search(line)
    out_file:write(result .. '\n')

    for n = 1, #nbests do
      out_file:write(nbests[n] .. '\n')
    end
  end

  print(string.format("PRED AVG SCORE: %.4f, PRED PPL: %.4f", pred_score_total / pred_words_total,
    math.exp(-pred_score_total/pred_words_total)))
  if opt.score_gold == 1 then
    print(string.format("GOLD AVG SCORE: %.4f, GOLD PPL: %.4f",
      gold_score_total / gold_words_total,
      math.exp(-gold_score_total/gold_words_total)))
  end
  out_file:close()
end

main()
