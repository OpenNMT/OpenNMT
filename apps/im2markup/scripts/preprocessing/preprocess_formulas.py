#!/usr/bin/env python
# tokenize latex formulas
import sys, os, argparse, logging, subprocess, shutil

def is_ascii(str):
    try:
        str.decode('ascii')
        return True
    except UnicodeError:
        return False

def process_args(args):
    parser = argparse.ArgumentParser(description='Preprocess (tokenize or normalize) latex formulas')

    parser.add_argument('--mode', dest='mode',
                        choices=['tokenize', 'normalize'], required=True,
                        help=('Tokenize (split to tokens seperated by space) or normalize (further translate to an equivalent standard form).'
                        ))
    parser.add_argument('--input-file', dest='input_file',
                        type=str, required=True,
                        help=('Input file containing latex formulas. One formula per line.'
                        ))
    parser.add_argument('--output-file', dest='output_file',
                        type=str, required=True,
                        help=('Output file.'
                        ))
    parser.add_argument('--num-threads', dest='num_threads',
                        type=int, default=4,
                        help=('Number of threads, default=4.'
                        ))
    parser.add_argument('--log-path', dest="log_path",
                        type=str, default='log.txt',
                        help=('Log file path, default=log.txt' 
                        ))
    parameters = parser.parse_args(args)
    return parameters

def main(args):
    parameters = process_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        filename=parameters.log_path)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info('Script being executed: %s'%__file__)

    input_file = parameters.input_file
    output_file = parameters.output_file

    assert os.path.exists(input_file), input_file
    cmd = "perl -pe 's|hskip(.*?)(cm\\|in\\|pt\\|mm\\|em)|hspace{\\1\\2}|g' %s > %s"%(input_file, output_file)
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        logging.error('FAILED: %s'%cmd)

    temp_file = output_file + '.tmp'
    with open(temp_file, 'w') as fout:
        fout.write(open(output_file).read().replace('\r', ' ')) # delete \r
    #shutil.copy(output_file, temp_file)

    cmd = "cat %s | node scripts/preprocessing/preprocess_latex.js %s > %s "%(temp_file, parameters.mode, output_file)
    ret = subprocess.call(cmd, shell=True)
    os.remove(temp_file)
    if ret != 0:
        logging.error('FAILED: %s'%cmd)
    temp_file = output_file + '.tmp'
    shutil.move(output_file, temp_file)
    with open(temp_file) as fin:
        with open(output_file, 'w') as fout:
            for line in fin:
                tokens = line.strip().split()
                tokens_out = []
                for token in tokens:
                    if is_ascii(token):
                        tokens_out.append(token)
                fout.write(' '.join(tokens_out)+'\n')
    os.remove(temp_file)

if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')
