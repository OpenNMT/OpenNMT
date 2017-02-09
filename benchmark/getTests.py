import requests
import argparse
import sys
import time
from datetime import date
import os
import json
import subprocess
import re

url = 'http://scorer.nmt-benchmark.net'

def main(args):
    scriptpath=os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(description='Benchmark test download')
    parser.add_argument('--apikey', help='Your API key',required=True)
    parser.add_argument('--src', help='Source language',required=True)
    parser.add_argument('--tgt', help='Target language',required=True)
    parser.add_argument('--basedir', default='.',
                        help='Directory to download the tests to - will create a directory')

    args = parser.parse_args(args)

    assert len(args.src)==2 and len(args.tgt)==2, "invalid langugage pair"
    assert len(args.apikey)==20, "invalid api key"
    assert os.path.isdir(args.basedir), "invalid directory"

    today = date.today()
    testdir = "benchmark-nmt-test-"+args.src+args.tgt+"-"+today.isoformat()
    pathtestdir = os.path.join(args.basedir,testdir)

    assert not os.path.exists(pathtestdir), "directory ["+pathtestdir+"] already exists"

    response = requests.get(url + '/test/list/' + args.apikey + '?src=' + args.src + '&tgt=' + args.tgt)
    if response.status_code != requests.codes.ok:
        response.raise_for_status()
        sys.exit(1)

    R=response.json()

    if len(R)==0:
        print 'no testset available for '+args.src+'>'+args.tgt
        sys.exit(0)
    else:
        print str(len(R))+' testset(s) available for '+args.src+'>'+args.tgt

    os.mkdir(pathtestdir)

    for r in R:
        print "downloading - "+r['source']['fileName']+" into "+pathtestdir
        response = requests.get(url + '/test/download/' + args.apikey + '?fileId=' + r['_id'])
        if response.status_code != requests.codes.ok:
            response.raise_for_status()
            os.exit(1)
        with open(os.path.join(pathtestdir,r['source']['fileName']+".info"), "wb") as testinfo:
            testinfo.write(json.dumps(r,indent=True))
        with open(os.path.join(pathtestdir,r['source']['fileName']+".org"), "wb") as test:
            test.write(response.content)
        testtxt = open(os.path.join(pathtestdir,r['source']['fileName']+".txt"), "wb")
        if re.search(r"\.(sgm|sgml|xml)$",r['source']['fileName']):
            p = subprocess.Popen(["perl",scriptpath+"/3rdParty/input-from-sgm.perl"],stdin=subprocess.PIPE,stdout=testtxt)
            p.stdin.write(response.content)
            p.communicate()
        else:
            testtxt.write(response.content)
        print "--> text version available: ",os.path.join(pathtestdir,r['source']['fileName']+".txt")
        testtxt.close()




if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
