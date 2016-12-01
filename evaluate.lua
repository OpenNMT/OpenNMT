require('./lib/eval')
require('./lib/utils')

local cmd = torch.CmdLine()

cmd:text("")
cmd:text("**evaluate.lua**")
cmd:text("")


cmd:option('-config', '', [[Read options from this file]])

cmd:text("")
cmd:text("**Data options**")
cmd:text("")

cmd:option('-model', '', [[Path to model .t7 file]])
cmd:option('-src_file', '', [[Source sequence to decode (one line per sequence)]])
cmd:option('-targ_file', '', [[True target sequence (optional)]])
cmd:option('-output_file', 'pred.txt', [[Path to output the predictions (each line will be the decoded sequence]])

-- beam search options
cmd:text("")
cmd:text("**Beam Search options**")
cmd:text("")
cmd:option('-beam', 5,[[Beam size]])
cmd:option('-batch', 30, [[Batch size]])
cmd:option('-max_sent_l', 250, [[Maximum sentence length. If any sequences in srcfile are longer than this then it will error out]])
cmd:option('-replace_unk', false, [[Replace the generated UNK tokens with the source token that
                              had the highest attention weight. If srctarg_dict is provided,
                              it will lookup the identified source token and give the corresponding
                              target token. If it is not provided (or the identified source token
                              does not exist in the table) then it will copy the source token]])
cmd:option('-srctarg_dict', '', [[Path to source-target dictionary to replace UNK
                                               tokens. See README.md for the format this file should be in]])
cmd:option('-n_best', 1, [[If > 1, it will also output an n_best list of decoded sentences]])

cmd:text("")
cmd:text("**Other options**")
cmd:text("")
cmd:option('-gpuid', -1, [[ID of the GPU to use (-1 = use CPU, 0 = let cuda choose between available GPUs)]])
cmd:option('-fallback_to_cpu', false, [[If = true, fallback to CPU if no GPU available]])
cmd:option('-cudnn', false, [[If using character model, this should be true if the character model was trained using cudnn]])
cmd:option('-time', false, [[Measure batch translation time]])


local function report_score(name, score_total, words_total)
  print(string.format(name .. " AVG SCORE: %.4f, " .. name .. " PPL: %.4f",
                      score_total / words_total,
                      math.exp(-score_total/words_total)))
end

local function main()
  local opt = cmd:parse(arg)

  local required_options = {
    "model",
    "src_file"
  }

  utils.Opt.init(opt, required_options)

  local src_reader = utils.FileReader.new(opt.src_file)
  local src_batch = {}
  local src_words_batch = {}
  local src_features_batch = {}

  local targ_reader
  local targ_batch
  local targ_words_batch
  local targ_features_batch

  local with_gold_score = opt.targ_file:len() > 0

  if with_gold_score then
    targ_reader = utils.FileReader.new(opt.targ_file)
    targ_batch = {}
    targ_words_batch = {}
    targ_features_batch = {}
  end

  eval.Translate.init(opt)

  local out_file = io.open(opt.output_file, 'w')

  local sent_id = 1
  local batch_id = 1

  local pred_score_total = 0
  local pred_words_total = 0
  local gold_score_total = 0
  local gold_words_total = 0

  local timer
  if opt.time then
    timer = torch.Timer()
    timer:stop()
    timer:reset()
  end

  while true do
    local src_tokens = src_reader:next()
    local targ_tokens
    if with_gold_score then
      targ_tokens = targ_reader:next()
    end

    if src_tokens ~= nil then
      local src_words, src_feats = utils.Features.extract(src_tokens)
      table.insert(src_batch, src_tokens)
      table.insert(src_words_batch, src_words)
      if #src_feats > 0 then
        table.insert(src_features_batch, src_feats)
      end

      if with_gold_score then
        local targ_words, targ_feats = utils.Features.extract(targ_tokens)
        table.insert(targ_batch, targ_tokens)
        table.insert(targ_words_batch, targ_words)
        if #targ_feats > 0 then
          table.insert(targ_features_batch, targ_feats)
        end
      end
    elseif #src_batch == 0 then
      break
    end

    if src_tokens == nil or #src_batch == opt.batch then
      if opt.time then
        timer:resume()
      end

      local pred_batch, info = eval.Translate.translate(src_words_batch, src_features_batch,
                                                        targ_words_batch, targ_features_batch)

      if opt.time then
        timer:stop()
      end

      for b = 1, #pred_batch do
        local src_sent = table.concat(src_batch[b], " ")
        local pred_sent = table.concat(pred_batch[b], " ")

        out_file:write(pred_sent .. '\n')

        print('SENT ' .. sent_id .. ': ' .. src_sent)
        print('PRED ' .. sent_id .. ': ' .. pred_sent)
        print(string.format("PRED SCORE: %.4f", info[b].score))

        pred_score_total = pred_score_total + info[b].score
        pred_words_total = pred_words_total + #pred_batch[b]

        if with_gold_score then
          local targ_sent = table.concat(targ_batch[b], " ")

          print('GOLD ' .. sent_id .. ': ' .. targ_sent)
          print(string.format("GOLD SCORE: %.4f", info[b].gold_score))

          gold_score_total = gold_score_total + info[b].gold_score
          gold_words_total = gold_words_total + #targ_batch[b]
        end

        if opt.n_best > 1 then
          print('\nBEST HYP:')
          for n = 1, #info[b].n_best do
            local n_best = table.concat(info[b].n_best[n].tokens, " ")
            print(string.format("[%.4f] %s", info[b].n_best[n].score, n_best))
          end
        end

        print('')
        sent_id = sent_id + 1
      end

      if src_tokens == nil then
        break
      end

      batch_id = batch_id + 1
      src_batch = {}
      src_words_batch = {}
      src_features_batch = {}
      if with_gold_score then
        targ_batch = {}
        targ_words_batch = {}
        targ_features_batch = {}
      end
      collectgarbage()
    end
  end

  if opt.time then
    local time = timer:time()
    local sentence_count = sent_id-1
    io.stderr:write("Average sentence translation time (in seconds):\n")
    io.stderr:write("avg real\t" .. time.real / sentence_count .. "\n")
    io.stderr:write("avg user\t" .. time.user / sentence_count .. "\n")
    io.stderr:write("avg sys\t" .. time.sys / sentence_count .. "\n")
  end

  report_score('PRED', pred_score_total, pred_words_total)

  if with_gold_score then
    report_score('GOLD', gold_score_total, gold_words_total)
  end

  out_file:close()
end

main()
