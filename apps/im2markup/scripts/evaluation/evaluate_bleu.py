import os, sys, copy, argparse, shutil, pickle, subprocess, logging

def process_args(args):
    parser = argparse.ArgumentParser(description='Evaluate BLEU score')
    parser.add_argument('--result-path', dest='result_path',
                        type=str, required=True,
                        help=('Result file containing <img_path> <label_gold> <label_pred> <score_pred> <score_gold> per line. This should be set to the output file of the model.'
                        ))
    parser.add_argument('--data-path', dest='data_path',
                        type=str, required=True,
                        help=('Input file which contains the samples to be evaluated. The format is <img_path> <label_idx> per line.'
                        ))
    parser.add_argument('--label-path', dest='label_path',
                        type=str, required=True,
                        help=('Gold label file which contains a tokenized formula per line.'
                        ))
    parser.add_argument('--log-path', dest="log_path",
                        type=str, default='log.txt',
                        help=('Log file path, default=log.txt' 
                        ))
    parameters = parser.parse_args(args)
    return parameters

def main(args):
    script_path = os.path.realpath(__file__)
    script_dir = os.path.dirname(script_path)
    app_dir = os.path.join(script_dir, '../..')

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
   
    label_path = parameters.label_path
    data_path = parameters.data_path
    result_path = parameters.result_path
    assert os.path.exists(label_path), 'Label file %s not found'%label_path
    assert os.path.exists(data_path), 'Data file %s not found'%data_path
    assert os.path.exists(result_path), 'Result file %s not found'%result_path

    labels_tmp = {}
    labels = {}
    with open(label_path) as flabel:
        with open(data_path) as fdata:
            line_idx = 0
            for line in flabel:
                labels_tmp[line_idx] = line.strip()
                line_idx += 1
            for line in fdata:
                img_path, idx = line.strip().split()
                labels[img_path] = labels_tmp[int(idx)]

    results = {}
    with open(result_path) as fin:
        for line_idx,line in enumerate(fin):
            if line_idx % 1000 == 0:
                print (line_idx)
            items = line.strip().split('\t')
            if len(items) == 5:
                img_path, label_gold, label_pred, score_pred, score_gold = items
                if not img_path in labels:
                    logging.warning('%s in result file while not in the gold file!'%img_path)
                results[img_path] = label_pred+'\n'

    fpred = open('.tmp.pred.txt', 'w')
    fgold = open('.tmp.gold.txt', 'w')
    for img_path in labels:
        fpred.write(results.setdefault(img_path, '\n'))
        fgold.write(labels[img_path]+'\n')
    fpred.close()
    fgold.close()
    metric = subprocess.check_output('perl third_party/multi-bleu.perl %s < %s'%('.tmp.gold.txt', '.tmp.pred.txt'), shell=True)
    #os.remove('.tmp.pred.txt')
    #os.remove('.tmp.gold.txt')
    logging.info(metric)

if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')
