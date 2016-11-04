require 'torch'
local lfs = require 'lfs'
local beam = require 's2sa.beam.main'
local Gold = require 's2sa.beam.gold'
local tokens = require 's2sa.beam.tokens'
local path = require 'pl.path'

local cmd = torch.CmdLine()

-- file location
cmd:option('-model', 'seq2seq_lstm_attn.t7.', [[Path to model .t7 file]])
cmd:option('-src_file', '', [[Source sequence to decode (one line per sequence)]])
cmd:option('-targ_file', '', [[True target sequence (optional)]])
cmd:option('-output_file', 'pred.txt', [[Path to output the predictions (each line will be the decoded sequence]])
cmd:option('-src_dict', '', [[Path to source vocabulary (*.src.dict file)]])
cmd:option('-targ_dict', '', [[Path to target vocabulary (*.targ.dict file)]])
cmd:option('-feature_dict_prefix', '', [[Prefix of the path to features vocabularies (*.feature_N.dict file)]])
cmd:option('-char_dict', '', [[If using chars, path to character vocabulary (*.char.dict file)]])

-- beam search options
cmd:option('-beam', 5,[[Beam size]])
cmd:option('-batch', 30, [[Batch size]])
cmd:option('-max_sent_l', 250, [[Maximum sentence length. If any sequences in srcfile are longer than this then it will error out]])
cmd:option('-simple', 0, [[If = 1, output prediction is simply the first time the top of the beam
                         ends with an end-of-sentence token. If = 0, the model considers all
                         hypotheses that have been generated so far that ends with end-of-sentence
                         token and takes the highest scoring of all of them.]])
cmd:option('-replace_unk', false, [[Replace the generated UNK tokens with the source token that
                              had the highest attention weight. If srctarg_dict is provided,
                              it will lookup the identified source token and give the corresponding
                              target token. If it is not provided (or the identified source token
                              does not exist in the table) then it will copy the source token]])
cmd:option('-srctarg_dict', '', [[Path to source-target dictionary to replace UNK
                                               tokens. See README.md for the format this file should be in]])
cmd:option('-score_gold', false, [[If = true, score the log likelihood of the gold as well]])
cmd:option('-n_best', 1, [[If > 1, it will also output an n_best list of decoded sentences]])
cmd:option('-gpuid', -1, [[ID of the GPU to use (-1 = use CPU, 0 = let cuda choose between available GPUs)]])
cmd:option('-gpuid2', -1, [[Second GPU ID, setting the second GPU ID assumes that the first one is set as well (gpuid > 0)]])
cmd:option('-fallback_to_cpu', false, [[If = true, fallback to CPU if no GPU available]])
cmd:option('-cudnn', 0, [[If using character model, this should be = 1 if the character model was trained using cudnn]])
cmd:option('-time', 0, [[If = 1, measure batch translation time]])
cmd:option('-float', 0, [[If = 1, convert the model to use float precision]])
cmd:option('-need_politeness_tag', 0, [[If = 1, specify that the models needs a special politeness token]])


local function main()
  local opt = cmd:parse(arg)
  assert(path.exists(opt.src_file), 'src_file does not exist')

  beam.init(opt, lfs.currentdir())

  local gold = Gold.new({
    score_gold = opt.score_gold,
    gold_file = opt.targ_file,
    batch_size = opt.batch
  })

  local file = io.open(opt.src_file, "r")
  local out_file = io.open(opt.output_file,'w')

  local sent_id = 1
  local batch_id = 1

  local pred_score_total = 0
  local pred_words_total = 0

  local src_sents = {}
  local src_tokens = {}

  local timer
  if opt.time > 0 then
    timer = torch.Timer()
    timer:stop()
    timer:reset()
  end

  while true do
    local line = file:read("*line")

    if line ~= nil then
      table.insert(src_sents, line)
      table.insert(src_tokens, tokens.from_sentence(line))
    elseif #src_sents == 0 then
      break
    end

    if line == nil or #src_sents == opt.batch then
      if opt.time > 0 then
        timer:resume()
      end

      gold.batch_id = batch_id
      local result = beam.search(src_tokens, gold)
      if opt.time > 0 then
        timer:stop()
      end
      local targ_tokens = result['pred_tokens_batch']
      local info = result['info_batch']

      for i = 1, #targ_tokens do
        print('SENT ' .. sent_id .. ': ' .. src_sents[i])
        local targ_sent = tokens.to_sentence(targ_tokens[i])
        print('PRED ' .. sent_id .. ': ' .. targ_sent)
        out_file:write(targ_sent .. '\n')

        pred_score_total = pred_score_total + info[i].pred_score
        pred_words_total = pred_words_total + info[i].pred_words

        for n = 1, #info[i].nbests do
          local nbest = tokens.to_sentence(info[i].nbests[n].tokens)
          local out_n = string.format("%d ||| %s ||| %.4f", n, nbest, info[i].nbests[n].score)
          print(out_n)
          out_file:write(nbest .. '\n')
        end

        print('')
        sent_id = sent_id + 1
      end

      if line == nil then
        break
      end

      batch_id = batch_id + 1
      src_sents = {}
      src_tokens = {}
      collectgarbage()
    end
  end

  if opt.time > 0 then
    local time = timer:time()
    local sentence_count = sent_id-1
    print("Average sentence translation time (in seconds):")
    io.stderr:write("avg real\t" .. time.real / sentence_count .. "\n")
    io.stderr:write("avg user\t" .. time.user / sentence_count .. "\n")
    io.stderr:write("avg sys\t" .. time.sys / sentence_count .. "\n")
  end

  print(string.format("PRED AVG SCORE: %.4f, PRED PPL: %.4f", pred_score_total / pred_words_total,
    math.exp(-pred_score_total/pred_words_total)))

  gold:log_results()

  out_file:close()
end

main()
