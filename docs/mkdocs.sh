cp _markdown/onmt/README.md tree/index.md

cd ../
mkdir -p doc/tree/options/
th preprocess.lua -h -md > doc/tree/options/preprocess.md
th train.lua -h -md > doc/tree/options/train.md
th translate.lua -h -md > doc/tree/options/translate.md
