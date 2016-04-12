#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create the data for the LSTM.
"""

import os
import sys
import argparse
import numpy as np
import h5py
import itertools
from collections import defaultdict

class Indexer:
    def __init__(self, symbols = ["*blank*","<unk>","<s>","</s>"]):
        self.vocab = defaultdict(int)
        self.PAD = symbols[0]
        self.UNK = symbols[1]
        self.BOS = symbols[2]
        self.EOS = symbols[3]
        self.d = {self.PAD: 1, self.UNK: 2, self.BOS: 3, self.EOS: 4}

    def add_w(self, ws):
        for w in ws:
            if w not in self.d:
                self.d[w] = len(self.d) + 1
            
    def convert(self, w):
        return self.d[w] if w in self.d else self.d[self.UNK]

    def convert_sequence(self, ls):
        return [self.convert(l) for l in ls]

    def clean(self, s):
        s = s.replace(self.PAD, "")
#        s = s.replace(self.UNK, "")
        s = s.replace(self.BOS, "")
        s = s.replace(self.EOS, "")
        return s
        
    def write(self, outfile):
        out = open(outfile, "w")
        items = [(v, k) for k, v in self.d.iteritems()]
        items.sort()
        for v, k in items:
            print >>out, k.encode('utf-8'), v
        out.close()

    def prune_vocab(self, k):
        vocab_list = [(word, count) for word, count in self.vocab.iteritems()]
        vocab_list.sort(key = lambda x: x[1], reverse=True)
        k = min(k, len(vocab_list))
        self.pruned_vocab = {pair[0]:pair[1] for pair in vocab_list[:k]}
        for word in self.pruned_vocab:
            if word not in self.d:
                self.d[word] = len(self.d) + 1

    def load_vocab(self, vocab_file):
        self.d = {}
        for line in open(vocab_file, 'r'):
            v, k = line.decode("utf-8").strip().split()
            self.d[v] = int(k)
            
def pad(ls, length, symbol):
    if len(ls) >= length:
        return ls[:length]
    return ls + [symbol] * (length -len(ls))
        
def get_data(args):
    src_indexer = Indexer(["<blank>","<unk>","<s>","</s>"])
    target_indexer = Indexer(["<blank>","<unk>","<s>","</s>"])    
    char_indexer = Indexer(["<blank>","<unk>","{","}"])
    char_indexer.add_w([src_indexer.PAD, src_indexer.UNK, src_indexer.BOS, src_indexer.EOS])
    
    def make_vocab(srcfile, targetfile, seqlength, max_word_l=0, chars=0):
        num_sents = 0
        for _, (src_orig, targ_orig) in \
                enumerate(itertools.izip(open(srcfile,'r'), open(targetfile,'r'))):
            src_orig = src_indexer.clean(src_orig.decode("utf-8").strip())
            targ_orig = target_indexer.clean(targ_orig.decode("utf-8").strip())
            targ = targ_orig.strip().split()
            src = src_orig.strip().split()
            if len(targ) > seqlength or len(src) > seqlength or len(targ) < 1 or len(src) < 1:
                continue
            num_sents += 1
            for word in targ:
                if chars == 1:
                    word = char_indexer.clean(word)
                    if len(word) == 0:
                        continue
                    max_word_l = max(len(word)+2, max_word_l)
                    for char in list(word):
                        char_indexer.vocab[char] += 1                                        
                target_indexer.vocab[word] += 1
                
            for word in src:
                if chars == 1:
                    word = char_indexer.clean(word)
                    if len(word) == 0:
                        continue
                    max_word_l = max(len(word)+2, max_word_l)
                    for char in list(word):
                        char_indexer.vocab[char] += 1                    
                src_indexer.vocab[word] += 1
                
        return max_word_l, num_sents
                
    def convert(srcfile, targetfile, batchsize, seqlength, outfile, num_sents,
                max_word_l, max_sent_l=0,chars=0, unkfilter=0):
        
        newseqlength = seqlength + 2 #add 2 for EOS and BOS
        targets = np.zeros((num_sents, newseqlength), dtype=int)
        target_output = np.zeros((num_sents, newseqlength), dtype=int)
        sources = np.zeros((num_sents, newseqlength), dtype=int)
        source_lengths = np.zeros((num_sents,), dtype=int)
        target_lengths = np.zeros((num_sents,), dtype=int)
        if chars==1:
            sources_char = np.zeros((num_sents, newseqlength, max_word_l), dtype=int)
            targets_char = np.zeros((num_sents, newseqlength, max_word_l), dtype=int)
        dropped = 0
        sent_id = 0
        for _, (src_orig, targ_orig) in \
                enumerate(itertools.izip(open(srcfile,'r'), open(targetfile,'r'))):
            src_orig = src_indexer.clean(src_orig.decode("utf-8").strip())
            targ_orig = target_indexer.clean(targ_orig.decode("utf-8").strip())
            targ = [target_indexer.BOS] + targ_orig.strip().split() + [target_indexer.EOS]
            src =  [src_indexer.BOS] + src_orig.strip().split() + [src_indexer.EOS]
            max_sent_l = max(len(targ), len(src), max_sent_l)
            if len(targ) > newseqlength or len(src) > newseqlength or len(targ) < 3 or len(src) < 3:
                dropped += 1
                continue                   
            targ = pad(targ, newseqlength+1, target_indexer.PAD)
            targ_char = []
            for word in targ:
                if chars == 1:
                    word = char_indexer.clean(word)
                #use UNK for target, but not for source
                word = word if word in target_indexer.d else target_indexer.UNK
                if chars == 1:
                    char = [char_indexer.BOS] + list(word) + [char_indexer.EOS]
                    if len(char) > max_word_l:
                        char = char[:max_word_l]
                        char[-1] = char_indexer.EOS
                    char_idx = char_indexer.convert_sequence(pad(char, max_word_l, char_indexer.PAD))
                    targ_char.append(char_idx)                    
            targ = target_indexer.convert_sequence(targ)
            targ = np.array(targ, dtype=int)

            src = pad(src, newseqlength, src_indexer.PAD)
            src_char = []
            for word in src:
                if chars == 1:
                    word = char_indexer.clean(word)
                    char = [char_indexer.BOS] + list(word) + [char_indexer.EOS]
                    if len(char) > max_word_l:
                        char = char[:max_word_l]
                        char[-1] = char_indexer.EOS
                    char_idx = char_indexer.convert_sequence(pad(char, max_word_l, char_indexer.PAD))
                    src_char.append(char_idx)
            src = src_indexer.convert_sequence(src)
            src = np.array(src, dtype=int)
            
            if unkfilter > 0:
                targ_unks = float((targ[:-1] == 2).sum())
                src_unks = float((src == 2).sum())                
                if unkfilter < 1: #unkfilter is a percentage if < 1
                    targ_unks = targ_unks/(len(targ[:-1])-2)
                    src_unks = src_unks/(len(src)-2)
                if targ_unks > unkfilter or src_unks > unkfilter:
                    dropped += 1
                    continue
                
            targets[sent_id] = np.array(targ[:-1],dtype=int)
            target_lengths[sent_id] = (targets[sent_id] != 1).sum()
            if chars == 1:
                targets_char[sent_id] = np.array(targ_char[:-1], dtype=int)
            target_output[sent_id] = np.array(targ[1:],dtype=int)                    
            sources[sent_id] = np.array(src, dtype=int)
            source_lengths[sent_id] = (sources[sent_id] != 1).sum()            
            if chars == 1:
                sources_char[sent_id] = np.array(src_char, dtype=int)

            sent_id += 1
            if sent_id % 100000 == 0:
                print("{}/{} sentences processed".format(sent_id, num_sents))

        print(sent_id, num_sents)
        #break up batches based on source lengths
        source_lengths = source_lengths[:sent_id]
        source_sort = np.argsort(source_lengths) 

        sources = sources[source_sort]
        targets = targets[source_sort]
        target_output = target_output[source_sort]
        target_l = target_lengths[source_sort]
        source_l = source_lengths[source_sort]

        curr_l = 0
        l_location = [] #idx where sent length changes
        
        for j,i in enumerate(source_sort):
            if source_lengths[i] > curr_l:
                curr_l = source_lengths[i]
                l_location.append(j+1)
        l_location.append(len(sources))

        #get batch sizes
        curr_idx = 1
        batch_idx = [1]
        nonzeros = []
        batch_l = []
        batch_w = []
        target_l_max = []
        for i in range(len(l_location)-1):
            while curr_idx < l_location[i+1]:
                curr_idx = min(curr_idx + batchsize, l_location[i+1])
                batch_idx.append(curr_idx)
        for i in range(len(batch_idx)-1):
            batch_l.append(batch_idx[i+1] - batch_idx[i])            
            batch_w.append(source_l[batch_idx[i]-1])
            nonzeros.append((target_output[batch_idx[i]-1:batch_idx[i+1]-1] != 1).sum().sum())
            target_l_max.append(max(target_l[batch_idx[i]-1:batch_idx[i+1]-1]))

        # Write output
        f = h5py.File(outfile, "w")
        
        f["source"] = sources
        f["target"] = targets
        f["target_output"] = target_output
        f["target_l"] = np.array(target_l_max, dtype=int)
        f["target_l_all"] = target_l        
        f["batch_l"] = np.array(batch_l, dtype=int)
        f["batch_w"] = np.array(batch_w, dtype=int)
        f["batch_idx"] = np.array(batch_idx[:-1], dtype=int)
        f["target_nonzeros"] = np.array(nonzeros, dtype=int)
        f["source_size"] = np.array([len(src_indexer.d)])
        f["target_size"] = np.array([len(target_indexer.d)])
        if chars == 1:            
            del sources, targets, target_output
            sources_char = sources_char[source_sort]
            f["source_char"] = sources_char
            del sources_char
            targets_char = targets_char[source_sort]                            
            f["target_char"] = targets_char
            f["char_size"] = np.array([len(char_indexer.d)])
        print("Saved {} sentences (dropped {} due to length/unk filter)".format(
            len(f["source"]), dropped))
        f.close()                
        return max_sent_l

    print("First pass through data to get vocab...")
    max_word_l, num_sents_train = make_vocab(args.srcfile, args.targetfile,
                                             args.seqlength, 0, args.chars)
    print("Number of sentences in training: {}".format(num_sents_train))
    max_word_l, num_sents_valid = make_vocab(args.srcvalfile, args.targetvalfile,
                                             args.seqlength, max_word_l, args.chars)
    print("Number of sentences in valid: {}".format(num_sents_valid))    
    if args.chars == 1:
        print("Max word length (before cutting): {}".format(max_word_l))
        max_word_l = min(max_word_l, args.maxwordlength)
        print("Max word length (after cutting): {}".format(max_word_l))

    #prune and write vocab
    src_indexer.prune_vocab(args.srcvocabsize)
    target_indexer.prune_vocab(args.targetvocabsize)
    if args.srcvocabfile != '':
        print('Loading pre-specified source vocab from ' + args.srcvocabfile)
        src_indexer.load_vocab(args.srcvocabfile)
    if args.targvocabfile != '':
        print('Loading pre-specified target vocab from ' + args.targvocabfile)
        target_indexer.load_vocab(args.targvocabfile)
        
    src_indexer.write(args.outputfile + ".src.dict")
    target_indexer.write(args.outputfile + ".targ.dict")
    if args.chars == 1:
        char_indexer.prune_vocab(200)
        char_indexer.write(args.outputfile + ".char.dict")
        print("Character vocab size: {}".format(len(char_indexer.pruned_vocab)))
    
    print("Source vocab size: Original = {}, Pruned = {}".format(len(src_indexer.vocab), 
                                                          len(src_indexer.d)))
    print("Target vocab size: Original = {}, Pruned = {}".format(len(target_indexer.vocab), 
                                                          len(target_indexer.d)))

    max_sent_l = 0
    max_sent_l = convert(args.srcvalfile, args.targetvalfile, args.batchsize, args.seqlength,
                         args.outputfile + "-val.hdf5", num_sents_valid,
                         max_word_l, max_sent_l, args.chars, args.unkfilter)
    max_sent_l = convert(args.srcfile, args.targetfile, args.batchsize, args.seqlength,
                         args.outputfile + "-train.hdf5", num_sents_train, max_word_l,
                         max_sent_l, args.chars, args.unkfilter)
    
    print("Max sent length (before dropping): {}".format(max_sent_l))    
    
def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--srcvocabsize', help="Source vocab size", type=int, default=50000)
    parser.add_argument('--targetvocabsize', help="Target vocab size", type=int, default=50000)
    parser.add_argument('--srcfile', help="Source Input file")
    parser.add_argument('--targetfile', help="Target Input file")
    parser.add_argument('--srcvalfile', help="Source Val file")
    parser.add_argument('--targetvalfile', help="Target val file")
    parser.add_argument('--batchsize', help="Batchsize", type=int, default=64)
    parser.add_argument('--seqlength', help="(Max) Sequence length", type=int, default=50)
    parser.add_argument('--outputfile', help="HDF5 output file", type=str)
    parser.add_argument('--maxwordlength', help="Max word length", type=int, default=35)
    parser.add_argument('--chars', help="Use characters", type=int, default=0)
    parser.add_argument('--srcvocabfile',
                         help="Source vocab file, if using pre-specified vocab",
                         type = str, default='')
    parser.add_argument('--targvocabfile',
                         help="Target vocab file, if using pre-specified vocab",
                         type = str, default='')
    parser.add_argument('--unkfilter',
                         help="Ignore sentences with too many <unk> tokens",
                         type = float, default = 0)
    args = parser.parse_args(arguments)
    get_data(args)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
