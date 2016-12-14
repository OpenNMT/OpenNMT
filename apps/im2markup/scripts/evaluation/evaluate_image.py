import sys, os, argparse, logging, glob
import numpy as np
from PIL import Image
import distance
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import difflib
from LevSeq import StringMatcher

def process_args(args):
    parser = argparse.ArgumentParser(description='Evaluate image related metrics.')

    parser.add_argument('--images-dir', dest='images_dir',
                        type=str, required=True,
                        help=('Images directory containing the rendered images. A subfolder with name "images_gold" for the rendered gold images, and a subfolder "images_pred" must be created beforehand by using scripts/evaluation/render_latex.py.'
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
   
    images_dir = parameters.images_dir
    gold_dir = os.path.join(images_dir, 'images_gold')
    pred_dir = os.path.join(images_dir, 'images_pred')
    assert os.path.exists(gold_dir), gold_dir 
    assert os.path.exists(pred_dir), pred_dir 
    total_edit_distance = 0
    total_ref = 0
    total_num = 0
    total_correct = 0
    total_correct_eliminate = 0
    filenames = glob.glob(os.path.join(gold_dir, '*'))
    for filename in filenames:
        filename2 = os.path.join(pred_dir, os.path.basename(filename))
        edit_distance, ref, match1, match2 = img_edit_distance_file(filename, filename2)
        total_edit_distance += edit_distance
        total_ref += ref
        total_num += 1
        if match1:
            total_correct += 1
        if match2:
            total_correct_eliminate += 1
        if total_num % 100 == 0:
            logging.info ('Total Num: %d'%total_num)
            logging.info ('Accuracy (w spaces): %f'%(float(total_correct)/total_num))
            logging.info ('Accuracy (w/o spaces): %f'%(float(total_correct_eliminate)/total_num))
            logging.info ('Edit Dist (w spaces): %f'%(1.-float(total_edit_distance)/total_ref))
            logging.info ('Total Correct (w spaces): %d'%total_correct)
            logging.info ('Total Correct (w/o spaces): %d'%total_correct_eliminate)
            logging.info ('Total Edit Dist (w spaces): %d'%total_edit_distance)
            logging.info ('Total Ref (w spaces): %d'%total_ref)
            logging.info ('')

    logging.info ('------------------------------------')
    logging.info ('Final')
    logging.info ('Total Num: %d'%total_num)
    logging.info ('Accuracy (w spaces): %f'%(float(total_correct)/total_num))
    logging.info ('Accuracy (w/o spaces): %f'%(float(total_correct_eliminate)/total_num))
    logging.info ('Edit Dist (w spaces): %f'%(1.-float(total_edit_distance)/total_ref))
    logging.info ('Total Correct (w spaces): %d'%total_correct)
    logging.info ('Total Correct (w/o spaces): %d'%total_correct_eliminate)
    logging.info ('Total Edit Dist (w spaces): %d'%total_edit_distance)
    logging.info ('Total Ref (w spaces): %d'%total_ref)

# return (edit_distance, ref, match, match w/o)
def img_edit_distance(im1, im2, out_path=None):
    img_data1 = np.asarray(im1, dtype=np.uint8) # height, width
    img_data1 = np.transpose(img_data1)
    h1 = img_data1.shape[1]
    w1 = img_data1.shape[0]
    img_data1 = (img_data1<=128).astype(np.uint8)
    if im2:
        img_data2 = np.asarray(im2, dtype=np.uint8) # height, width
        img_data2 = np.transpose(img_data2)
        h2 = img_data2.shape[1]
        w2 = img_data2.shape[0]
        img_data2 = (img_data2<=128).astype(np.uint8)
    else:
        img_data2 = []
        h2 = h1
    if h1 == h2:
        seq1 = [''.join([str(i) for i in item]) for item in img_data1]
        seq2 = [''.join([str(i) for i in item]) for item in img_data2]
    elif h1 > h2:# pad h2
        seq1 = [''.join([str(i) for i in item]) for item in img_data1]
        seq2 = [''.join([str(i) for i in item])+''.join(['0']*(h1-h2)) for item in img_data2]
    else:
        seq1 = [''.join([str(i) for i in item])+''.join(['0']*(h2-h1)) for item in img_data1]
        seq2 = [''.join([str(i) for i in item]) for item in img_data2]

    seq1_int = [int(item,2) for item in seq1]
    seq2_int = [int(item,2) for item in seq2]
    big = int(''.join(['0' for i in range(max(h1,h2))]),2)
    seq1_eliminate = []
    seq2_eliminate = []
    seq1_new = []
    seq2_new = []
    for idx,items in enumerate(seq1_int):
        if items>big:
            seq1_eliminate.append(items)
            seq1_new.append(seq1[idx])
    for idx,items in enumerate(seq2_int):
        if items>big:
            seq2_eliminate.append(items)
            seq2_new.append(seq2[idx])
    if len(seq2) == 0:
        return (len(seq1), len(seq1), False, False)

    def make_strs(int_ls, int_ls2):
        d = {}
        seen = []
        def build(ls):
            for l in ls:
                if int(l, 2) in d: continue
                found = False
                l_arr = np.array(map(int, l))
            
                for l2,l2_arr in seen:
                    if np.abs(l_arr -l2_arr).sum() < 5:
                        d[int(l, 2)] = d[int(l2, 2)]
                        found = True
                        break
                if not found:
                    d[int(l, 2)] = unichr(len(seen))
                    seen.append((l, np.array(map(int, l))))
                    
        build(int_ls)
        build(int_ls2)
        return "".join([d[int(l, 2)] for l in int_ls]), "".join([d[int(l, 2)] for l in int_ls2])
    #if out_path:
    seq1_t, seq2_t = make_strs(seq1, seq2)

    edit_distance = distance.levenshtein(seq1_int, seq2_int)
    match = True
    if edit_distance>0:
        matcher = StringMatcher(None, seq1_t, seq2_t)

        ls = []
        for op in matcher.get_opcodes():
            if op[0] == "equal" or (op[2]-op[1] < 5):
                ls += [[int(r) for r in l]
                       for l in seq1[op[1]:op[2]]
                       ] 
            elif op[0] == "replace":
                a = seq1[op[1]:op[2]]
                b = seq2[op[3]:op[4]]
                ls += [[int(r1)*3 + int(r2)*2
                        if int(r1) != int(r2) else int(r1)
                        for r1, r2 in zip(a[i] if i < len(a) else [0]*1000,
                                          b[i] if i < len(b) else [0]*1000)]
                       for i in range(max(len(a), len(b)))]
                match = False
            elif op[0] == "insert":

                ls += [[int(r)*3 for r in l]
                       for l in seq2[op[3]:op[4]]]
                match = False
            elif op[0] == "delete":
                match = False
                ls += [[int(r)*2 for r in l] for l in seq1[op[1]:op[2]]]

        #vmax = 3
        #plt.imshow(np.array(ls).transpose(), vmax=vmax)

        #cmap = LinearSegmentedColormap.from_list('mycmap', [(0. /vmax, 'white'),
        #                                                    (1. /vmax, 'grey'),
        #                                                    (2. /vmax, 'blue'),
        #                                                    (3. /vmax, 'red')])

        #plt.set_cmap(cmap)
        #plt.axis('off')
        #plt.savefig(out_path, bbox_inches="tight")

    match1 = match
    seq1_t, seq2_t = make_strs(seq1_new, seq2_new)

    if len(seq2_new) == 0 or len(seq1_new) == 0:
        if len(seq2_new) == len(seq1_new):
            return (edit_distance, max(len(seq1_int),len(seq2_int)), match1, True)# all blank
        return (edit_distance, max(len(seq1_int),len(seq2_int)), match1, False)
    match = True
    matcher = StringMatcher(None, seq1_t, seq2_t)

    ls = []
    for op in matcher.get_opcodes():
        if op[0] == "equal" or (op[2]-op[1] < 5):
            ls += [[int(r) for r in l]
                   for l in seq1[op[1]:op[2]]
                   ] 
        elif op[0] == "replace":
            a = seq1[op[1]:op[2]]
            b = seq2[op[3]:op[4]]
            ls += [[int(r1)*3 + int(r2)*2
                    if int(r1) != int(r2) else int(r1)
                    for r1, r2 in zip(a[i] if i < len(a) else [0]*1000,
                                      b[i] if i < len(b) else [0]*1000)]
                   for i in range(max(len(a), len(b)))]
            match = False
        elif op[0] == "insert":

            ls += [[int(r)*3 for r in l]
                   for l in seq2[op[3]:op[4]]]
            match = False
        elif op[0] == "delete":
            match = False
            ls += [[int(r)*2 for r in l] for l in seq1[op[1]:op[2]]]

    match2 = match

    return (edit_distance, max(len(seq1_int),len(seq2_int)), match1, match2)

def img_edit_distance_file(file1, file2, output_path=None):
    img1 = Image.open(file1).convert('L')
    if os.path.exists(file2):
        img2 = Image.open(file2).convert('L')
    else:
        img2 = None
    return img_edit_distance(img1, img2, output_path)

if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')
