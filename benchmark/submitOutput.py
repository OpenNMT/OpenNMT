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
    parser.add_argument('--systemId', help='Id of the system',required=True)
    parser.add_argument('--info', help='info file',required=True)
    parser.add_argument('--output', help='output file',required=True)

    args = parser.parse_args(args)

    assert len(args.apikey)==20, "invalid api key"
    assert len(args.systemId)==24, "invalid system id"

    assert os.path.exists(args.info), "info file does not exist"

    infojson = None
    with open(args.info) as infofile:
        infojson = json.load(infofile)

    testId = infojson["_id"]

    sourcefile=re.sub(r".info",".org",args.info)

    output = None
    print "* read output file"
    with open(args.output) as f:
        output=f.read()

    if re.search(r"\.(sgm|sgml|xml)$",infojson['source']['fileName']):
        p = subprocess.Popen(["perl",scriptpath+"/3rdParty/wrap-xml.perl",
                                infojson['target']['language'],
                                sourcefile, "NMT"],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
        print "* convert output to sgm file"
        p.stdin.write(output)
        output=p.communicate()[0]

    print "* save '"+re.sub(r".info",".trans",args.info)+"'"
    with open(re.sub(r".info",".trans",args.info),"w") as fOutput:
        fOutput.write(output)
        fOutput.close()

    data = {'systemId': args.systemId, 'fileId': testId}
    files = {'outputFile': open(re.sub(r".info",".trans",args.info),"r")}
    response = requests.post(url + '/output/upload/' + args.apikey, data=data, files=files)
    if response.status_code != requests.codes.ok:
        response.raise_for_status()
    else:
        print(response)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
