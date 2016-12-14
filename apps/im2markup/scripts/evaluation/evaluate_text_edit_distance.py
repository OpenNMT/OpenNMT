import os, sys, argparse, logging
import distance


def process_args(args):
    parser = argparse.ArgumentParser(description='Evaluate text edit distance.')

    parser.add_argument('--result-path', dest='result_path',
                        type=str, required=True,
                        help=('Result file containing <img_path> <label_gold> <label_pred> <score_pred> <score_gold> per line. This should be set to the output file of the model.'
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

    result_file = parameters.result_path
    total_ref = 0
    total_edit_distance = 0
    with open(result_file) as fin:
        for idx,line in enumerate(fin):
            if idx % 100 == 0:
                print (idx)
            items = line.strip().split('\t')
            if len(items) == 5:
                img_path, label_gold, label_pred, score_pred, score_gold = items
                l_pred = label_pred.strip()
                l_gold = label_gold.strip()
                tokens_pred = l_pred.split(' ')
                tokens_gold = l_gold.split(' ')
                ref = max(len(tokens_gold), len(tokens_pred))
                edit_distance = distance.levenshtein(tokens_gold, tokens_pred)
                total_ref += ref
                total_edit_distance += edit_distance
    logging.info('Edit Distance Accuracy: %f'%(1.-float(total_edit_distance)/total_ref))
   
if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')
