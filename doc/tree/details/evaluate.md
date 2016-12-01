
**evaluate.lua**


config
:   Read options from this file []


**Data options**


model
:   Path to model .t7 file []

src_file
:   Source sequence to decode (one line per sequence) []

targ_file
:   True target sequence (optional) []

output_file
:   Path to output the predictions (each line will be the decoded sequence [pred.txt]


**Beam Search options**


beam
:   Beam size [5]

batch
:   Batch size [30]

max_sent_l
:   Maximum sentence length. If any sequences in srcfile are longer than this then it will error out [250]

replace_unk
:   Replace the generated UNK tokens with the source token thathad the highest attention weight. If srctarg_dict is provided,it will lookup the identified source token and give the correspondingtarget token. If it is not provided (or the identified source tokendoes not exist in the table) then it will copy the source token [false]

srctarg_dict
:   Path to source-target dictionary to replace UNKtokens. See README.md for the format this file should be in []

n_best
:   If > 1, it will also output an n_best list of decoded sentences [1]


**Other options**


gpuid
:   ID of the GPU to use (-1 = use CPU, 0 = let cuda choose between available GPUs) [-1]

fallback_to_cpu
:   If = true, fallback to CPU if no GPU available [false]

cudnn
:   If using character model, this should be true if the character model was trained using cudnn [false]

