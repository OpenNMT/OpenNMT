#! /bin/sh
th preprocess.lua -h -md > docs/options/preprocess.md
th train.lua -h -md > docs/options/train.md
th translate.lua -h -md > docs/options/translate.md
