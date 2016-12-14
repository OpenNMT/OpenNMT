 --[[ Create LSTM unit, adapted from https://github.com/karpathy/char-rnn/blob/master/model/LSTM.lua
 -- and https://github.com/harvardnlp/seq2seq-attn/blob/master/models.lua
 --    ARGS:
 --        - `input_size`      : integer, number of input dimensions. If use_lookup is true, this is the embedding size
 --        - `num_hidden`  : integer, number of hidden nodes
 --        - `num_layers`  : integer, number of layers
 --        - `dropout`  : boolean, if true apply dropout
 --        - `use_attention`  : boolean, use attention or not (note that context size must be equal to num_hidden)
 --        - `input_feed`  : boolean, use input feeding approach or not
 --        - `use_lookup`  : boolean, use lookup table or not
 --        - `vocab_size`  : integer, vocabulary size
 --    RETURNS:
 --        - `LSTM` : constructed LSTM unit (nngraph module)
 --        inputs: x, (context), (prev attention), [prev_c, prev_h]*L
 --        outputs: [next_c, next_h]*L, h_out
 --]]
 
function createLSTM(input_size, num_hidden, num_layers,
    dropout,
    use_attention, input_feed,
    use_lookup, vocab_size,
    batch_size,
    max_encoder_l,
    model)

  dropout = dropout or 0 

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  local offset = 0
  if use_attention then
      table.insert(inputs, nn.Identity()()) -- all context (batch_size x source_l x num_hidden)
      offset = offset + 1
      if input_feed then
          table.insert(inputs, nn.Identity()()) -- prev context_attn (batch_size x num_hidden)
          offset = offset + 1
      end
  end
  for L = 1, num_layers do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1, num_layers do
    local nameL = model..'_L'..L..'_'
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1+offset]
    local prev_c = inputs[L*2+offset]
    -- the input to this layer
    if L == 1 then
      if use_lookup then
          local embeddings = nn.LookupTable(vocab_size, input_size)
          x = embeddings(inputs[1])
      else
          x = inputs[1]
      end
      input_size_L = input_size
      if input_feed then
          x = nn.JoinTable(2):usePrealloc("dec_inputfeed_join",
            {{batch_size, input_size},{batch_size, num_hidden}})({x, inputs[1+offset]}) -- batch_size x (input_size + num_hidden)
          input_size_L = input_size + num_hidden
      end    
    else 
      x = outputs[(L-1)*2] 
      if dropout then x = nn.Dropout(dropout):usePrealloc(nameL.."dropout",
          {{batch_size, input_size_L}})(x) end -- apply dropout, if any
      input_size_L = num_hidden
    end
    local i2h_name
    if input_size_L <= 4*num_hidden then
        i2h_name = 'i2h-reuse'
    else
        i2h_name = 'i2h'
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * num_hidden):usePrealloc(nameL..i2h_name,
                                                                      {{batch_size, input_size_L}},
                                                                      {{batch_size, 4*num_hidden}})(x)
    local h2h = nn.Linear(num_hidden, 4 * num_hidden):usePrealloc(nameL.."h2h-reuse",
                                                                        {{batch_size, num_hidden}},
                                                                        {{batch_size, 4*num_hidden}})(prev_h)
    local all_input_sums = nn.CAddTable():usePrealloc(nameL.."allinput",
                                                          {{batch_size, 4*num_hidden},{batch_size, 4*num_hidden}},
                                                          {{batch_size, 4*num_hidden}})({i2h, h2h})


    local reshaped = nn.Reshape(4, num_hidden)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2):usePrealloc(nameL.."reshapesplit",
                                                          {{batch_size, 4, num_hidden}})(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid():usePrealloc(nameL.."G1-reuse",{{batch_size, num_hidden}})(n1)
    local forget_gate = nn.Sigmoid():usePrealloc(nameL.."G2-reuse",{{batch_size, num_hidden}})(n2)
    local out_gate = nn.Sigmoid():usePrealloc(nameL.."G3-reuse",{{batch_size, num_hidden}})(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh():usePrealloc(nameL.."G4-reuse",{{batch_size, num_hidden}})(n4)
    -- perform the LSTM update
    local next_c           = nn.CAddTable():usePrealloc(nameL.."G5a",{{batch_size,num_hidden},{batch_size,num_hidden}})({
        nn.CMulTable():usePrealloc(nameL.."G5b",{{batch_size,num_hidden},{batch_size,num_hidden}})({forget_gate, prev_c}),
        nn.CMulTable():usePrealloc(nameL.."G5c",{{batch_size,num_hidden},{batch_size,num_hidden}})({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable():usePrealloc(nameL.."G5d",{{batch_size,num_hidden},{batch_size,num_hidden}})({out_gate, nn.Tanh():usePrealloc(nameL.."G6-reuse",{{batch_size,num_hidden}})(next_c)})
    
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end
  if use_attention then
    local top_h = outputs[#outputs]
    local decoder_out
    local decoder_attn = create_decoder_attn(num_hidden, 0, batch_size, max_encoder_l)
    decoder_attn.name = 'decoder_attn'
    decoder_out = decoder_attn({top_h, inputs[2]})
    if dropout then
      decoder_out = nn.Dropout(dropout, nil, false):usePrealloc("dec_dropout",{{batch_size,num_hidden}})(decoder_out)
    end     
    table.insert(outputs, decoder_out)
  end
  return nn.gModule(inputs, outputs)
end

function create_decoder_attn(num_hidden, simple, batch_size, max_encoder_l)
  -- inputs[1]: 2D tensor target_t (batch_l x num_hidden) and
  -- inputs[2]: 3D tensor for context (batch_l x source_l x input_size)
  
  local inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())
  local target_t = nn.LinearNoBias(num_hidden, num_hidden)(inputs[1])
  local context = inputs[2]
  simple = simple or 0
  -- get attention
  local attn = nn.MM():usePrealloc("dec_attn_mm1",
                                     {{batch_size, max_encoder_l, num_hidden},{batch_size, num_hidden, 1}},
                                     {{batch_size, max_encoder_l, 1}})({context, nn.Replicate(1,3)(target_t)}) -- batch_l x source_l x 1
  attn = nn.Sum(3)(attn)
  local softmax_attn = nn.SoftMax()
  softmax_attn.name = 'softmax_attn'
  attn = softmax_attn(attn)
  attn = nn.Replicate(1,2)(attn) -- batch_l x  1 x source_l
  
  -- apply attention to context
  local context_combined = nn.MM():usePrealloc("dec_attn_mm2",
                                                 {{batch_size, 1, max_encoder_l},{batch_size, max_encoder_l, num_hidden}},
                                                 {{batch_size, 1, num_hidden}})({attn, context}) -- batch_l x source_l x num_hidden
  context_combined = nn.Sum(2):usePrealloc("dec_attn_sum",
                                             {{batch_size, 1, num_hidden}},
                                             {{batch_size, num_hidden}})(context_combined) -- batch_l x num_hidden
  local context_output
  if simple == 0 then
    context_combined = nn.JoinTable(2):usePrealloc("dec_attn_jointable",
                            {{batch_size,num_hidden},{batch_size, num_hidden}})({context_combined, inputs[1]}) -- batch_l x num_hidden*2
    context_output = nn.Tanh():usePrealloc("dec_noattn_tanh",{{batch_size,num_hidden}})(nn.LinearNoBias(num_hidden*2, num_hidden):usePrealloc("dec_noattn_linear",
    {{batch_size,2*num_hidden}})(context_combined))
  else
    context_output = nn.CAddTable():usePrealloc("dec_attn_caddtable1",
    {{batch_size, num_hidden}, {batch_size, num_hidden}})({context_combined,inputs[1]})
  end   
  return nn.gModule(inputs, {context_output})   
end
