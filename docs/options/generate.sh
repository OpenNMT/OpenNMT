#! /bin/sh

gen_script_options ()
{
    echo "<!--- This file was automatically generated. Do not modify it manually but use the docs/options/generate.sh script instead. -->" > $2
    echo "" >> $2
    th $1 -h -md >> $2
}

gen_script_options preprocess.lua docs/options/preprocess.md
gen_script_options train.lua docs/options/train.md
gen_script_options translate.lua docs/options/translate.md
gen_script_options tag.lua docs/options/tag.md
gen_script_options tools/build_vocab.lua docs/options/build_vocab.md
gen_script_options tools/release_model.lua docs/options/release_model.md
gen_script_options tools/tokenize.lua docs/options/tokenize.md
gen_script_options tools/learn_bpe.lua docs/options/learn_bpe.md
gen_script_options tools/translation_server.lua docs/options/server.md
gen_script_options tools/rest_translation_server.lua docs/options/rest_server.md
gen_script_options tools/embeddings.lua docs/options/embeddings.md
