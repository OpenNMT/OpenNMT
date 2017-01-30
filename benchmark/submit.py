import requests
import argparse
import sys

url = 'http://localhost:3000'

def main(args):
    parser = argparse.ArgumentParser(description='Benchmark system submission')
    parser.add_argument('--apikey', help='Your API key')
    parser.add_argument('metadata', help='System metadata in JSON format')
    args = parser.parse_args(args)

    files = {'file': open(args.metadata, 'r')}
    response = requests.post(url + '/system/upload/' + args.apikey, files=files)
    if response.status_code != requests.codes.ok:
        response.raise_for_status()
    else:
        system_id = response.json()['_id']
        print(system_id)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
