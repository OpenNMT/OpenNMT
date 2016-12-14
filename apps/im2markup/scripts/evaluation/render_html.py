import sys, os, re, shutil, argparse, logging
sys.path.insert(0, '%s'%os.path.join(os.path.dirname(__file__), '../utils/'))
from image_utils import *
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool 

W=100
H=100

def process_args(args):
    parser = argparse.ArgumentParser(description='Render HTML files for comparison. Note that we render both the predicted results, and the original HTMLs.')

    parser.add_argument('--result-path', dest='result_path',
                        type=str, required=True,
                        help=('Result file containing <img_path> <label_gold> <label_pred> <score_pred> <score_gold> per line. This should be set to the output file of the model.'
                        ))
    parser.add_argument('--output-dir', dest='output_dir',
                        type=str, required=True,
                        help=('Output directory to put the rendered images. A subfolder with name "images_gold" will be created for the rendered gold images, and a subfolder with name "images_pred" will be created for the rendered predictions.'
                        ))

    parser.add_argument('--replace', dest='replace', action='store_true',
                        help=('Replace flag, if set to false, will ignore the already existing images.'
                        ))
    parser.add_argument('--no-replace', dest='replace', action='store_false')
    parser.set_defaults(replace=False)
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
   
    result_path = parameters.result_path
    output_dir = parameters.output_dir
    assert os.path.exists(result_path), result_path

    pred_dir = os.path.join(output_dir, 'images_pred')
    gold_dir = os.path.join(output_dir, 'images_gold')
    for dirname in [pred_dir, gold_dir]:
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    lines = []
    with open(result_path) as fin:
        for idx,line in enumerate(fin):
            items = line.strip().split('\t')
            if len(items) == 5:
                img_path, label_gold, label_pred, score_pred, score_gold = items
                img_idx = img_path[:-9]
                lines.append((label_pred, img_idx, pred_dir, parameters.replace))
                lines.append((label_gold, img_idx, gold_dir, parameters.replace))
    
    logging.info('Creating pool with %d threads'%parameters.num_threads)
    pool = ThreadPool(parameters.num_threads)
    logging.info('Jobs running...')
    results = pool.map(main_parallel, lines)
    pool.close() 
    pool.join() 


def main_parallel(l):
    label, img_idx, dirname, replace = l
    if replace or (not os.path.exists('%s/%s-full.png'%(dirname, img_idx))):
        html_name = '%s_%s.html'%(dirname, img_idx)
        with open(html_name, 'w') as fout:
            fout.write(label)
        os.system('webkit2png --clipwidth=1 --clipheight=1 -Fs 1 -W %d -H %d %s -o %s/%s'%(W,H,html_name,dirname,img_idx))
        os.remove(html_name)

if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')
