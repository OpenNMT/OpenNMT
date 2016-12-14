import sys, os, re, shutil, argparse, logging
sys.path.insert(0, '%s'%os.path.join(os.path.dirname(__file__), '../utils/'))
from utils import run
from image_utils import *
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool 


TIMEOUT = 10

# replace \pmatrix with \begin{pmatrix}\end{pmatrix}
# replace \matrix with \begin{matrix}\end{matrix}
template = r"""
\documentclass[12pt]{article}
\pagestyle{empty}
\usepackage{amsmath}
\newcommand{\mymatrix}[1]{\begin{matrix}#1\end{matrix}}
\newcommand{\mypmatrix}[1]{\begin{pmatrix}#1\end{pmatrix}}
\begin{document}
\begin{displaymath}
%s
\end{displaymath}
\end{document}
"""


def process_args(args):
    parser = argparse.ArgumentParser(description='Render latex formulas for comparison. Note that we need to render both the predicted results, and the original formulas, since we need to make sure the same environment of rendering is used.')

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
                        help=('Gold label file which contains a formula per line. Note that this does not necessarily need to be tokenized, and for comparing against the gold standard, the original (un-preprocessed) label file shall be used.'
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
    data_path = parameters.data_path
    label_path = parameters.label_path
    output_dir = parameters.output_dir
    assert os.path.exists(label_path), label_path
    assert os.path.exists(result_path), result_path
    assert os.path.exists(data_path), data_path

    pred_dir = os.path.join(output_dir, 'images_pred')
    gold_dir = os.path.join(output_dir, 'images_gold')
    for dirname in [pred_dir, gold_dir]:
        if not os.path.exists(dirname):
            os.makedirs(dirname)


    formulas = open(label_path).readlines()
    lines = []
    with open(data_path) as fin:
        for line in fin:
            img_path, line_idx = line.strip().split()
            lines.append((img_path, formulas[int(line_idx)], os.path.join(gold_dir, img_path), parameters.replace))
    with open(result_path) as fin:
        for line in fin:
            img_path, label_gold, label_pred, _, _ = line.strip().split('\t')
            lines.append((img_path, label_pred, os.path.join(pred_dir, img_path), parameters.replace))
    logging.info('Creating pool with %d threads'%parameters.num_threads)
    pool = ThreadPool(parameters.num_threads)
    logging.info('Jobs running...')
    results = pool.map(main_parallel, lines)
    pool.close() 
    pool.join() 

def output_err(output_path, i, reason, img):
    logging.info('ERROR: %s %s\n'%(img,reason))

def main_parallel(line):
    img_path, l, output_path, replace = line
    pre_name = output_path.replace('/', '_').replace('.','_')
    l = l.strip()
    l = l.replace(r'\pmatrix', r'\mypmatrix')
    l = l.replace(r'\matrix', r'\mymatrix')
    # remove leading comments
    l = l.strip('%')
    if len(l) == 0:
        l = '\\hspace{1cm}'
    # \hspace {1 . 5 cm} -> \hspace {1.5cm}
    for space in ["hspace", "vspace"]:
        match = re.finditer(space + " {(.*?)}", l)
        if match:
            new_l = ""
            last = 0
            for m in match:
                new_l = new_l + l[last:m.start(1)] + m.group(1).replace(" ", "")
                last = m.end(1)
            new_l = new_l + l[last:]
            l = new_l    
    if replace or (not os.path.exists(output_path)):
        tex_filename = pre_name+'.tex'
        log_filename = pre_name+'.log'
        aux_filename = pre_name+'.aux'
        with open(tex_filename, "w") as w: 
            print >> w, (template%l)
        run("pdflatex -interaction=nonstopmode %s  >/dev/null"%tex_filename, TIMEOUT)
        os.remove(tex_filename)
        os.remove(log_filename)
        os.remove(aux_filename)
        pdf_filename = tex_filename[:-4]+'.pdf'
        png_filename = tex_filename[:-4]+'.png'
        if not os.path.exists(pdf_filename):
            output_err(output_path, 0, 'cannot compile', img_path)
        else:
            os.system("convert -density 200 -quality 100 %s %s"%(pdf_filename, png_filename))
            os.remove(pdf_filename)
            if os.path.exists(png_filename):
                crop_image(png_filename, output_path)
                os.remove(png_filename)

        
if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')
