rm -fr tree/
mkdir tree/
cp _markdown/onmt/README.md tree/index.md
cp _markdown/onmt/Quickstart.md tree/Quickstart.md


mkdir tree/code/
mkdir tree/code/onmt
mkdir tree/code/train
mkdir tree/code/eval
mkdir tree/code/data

cp _markdown/onmt/lib+onmt+*.md tree/code/onmt/
mv tree/code/onmt/lib+onmt+init.md tree/code/onmt/index.md

cp _markdown/onmt/lib+train+*.md tree/code/train/
mv tree/code/train/lib+train+init.md tree/code/train/index.md
cp _markdown/onmt/lib+data.md tree/code/train/

cp _markdown/onmt/lib+eval+*.md tree/code/eval/
mv tree/code/eval/lib+eval+init.md tree/code/eval/index.md
rm tree/code/eval/lib+eval+translate.md 

rm -fr tree/details
mkdir tree/details/
cd ../
th preprocess.lua --help | python doc/format.py >> doc/tree/details/preprocess.md
th train.lua --help | python doc/format.py >> doc/tree/details/train.md
th translate.lua --help | python doc/format.py >> doc/tree/details/translate.md

