#-*- coding: utf-8 -*-

import subprocess
import os
import sys

# check if tree-tagger-flush exists

def start_model(path,m):
  global treetagger
  global nbuf
  try:
    treetagger = subprocess.Popen([path+'/tree-tagger-flush', m], stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=open(os.devnull, 'w'))
    nbuf = 10
  except:
    sys.stderr.write('Cannot find tree-tagger-flush, use tree-tagger: it will be less efficient\n')
    treetagger = subprocess.Popen([path+'/tree-tagger', '/Users/senellart/Downloads/french.par'], stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=open(os.devnull, 'w'))
    # this parameter should be adjusted for each os - it forces tree-tagger to flush by following this many sentence ends
    nbuf = 3000

extraneous = 0

def tag(s):
  global extraneous
  l=s.split()
  for w in l:
    treetagger.stdin.write(w+'\n')
  treetagger.stdin.write('\n')
  for _ in range(nbuf):
    treetagger.stdin.write('.\n')
  result = []
  for tag in treetagger.stdout:
    tag = tag.strip()
    if tag != '':
      if extraneous == 0:
        result.append(tag)
      else:
        extraneous -= 1
      if len(result)==len(l):
        break
  extraneous = nbuf
  return " ".join(result)

from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer
from SocketServer import ThreadingMixIn
import threading
import argparse
import re
import cgi

class LocalData(object):
  records = {}

class HTTPRequestHandler(BaseHTTPRequestHandler):
  def do_POST(self):
    if None != re.search('/pos', self.path):
      length = int(self.headers.getheader('content-length'))
      data = cgi.parse_qs(self.rfile.read(length), keep_blank_values=1)
      print("received sentence %s" % data)
      if 'sent' in data:
        self.send_response(200)
        self.end_headers()
        self.wfile.write(tag(data['sent'][0]))
      else:
        self.send_response(403)
        self.end_headers()
    return

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
  allow_reuse_address = True

  def shutdown(self):
    self.socket.close()
    HTTPServer.shutdown(self)

class SimpleHttpServer():
  def __init__(self, ip, port):
    self.server = ThreadedHTTPServer((ip,port), HTTPRequestHandler)

  def start(self):
    self.server_thread = threading.Thread(target=self.server.serve_forever)
    self.server_thread.daemon = True
    self.server_thread.start()

  def waitForThread(self):
    self.server_thread.join()

  def addRecord(self, recordID, jsonEncodedRecord):
    LocalData.records[recordID] = jsonEncodedRecord

  def stop(self):
    self.server.shutdown()
    self.waitForThread()

if __name__=='__main__':
  parser = argparse.ArgumentParser(description='TreeBank Wrapper')
  parser.add_argument('-sent', type=str, help='Test commandline mode - pass the sentence to tag')
  parser.add_argument('-port', default=3000, type=int, help='Listening port for HTTP Server')
  parser.add_argument('-ip', default="localhost", help='HTTP Server IP')
  parser.add_argument('-model', type=str, help='model to serve')
  parser.add_argument('-path', type=str, help='path to tree-tagger binaries')
  args = parser.parse_args()

  start_model(args.path, args.model)

  if args.sent and len(args.sent):
    print(tag(args.sent))
  else:
    server = SimpleHttpServer(args.ip, args.port)
    print('HTTP Server Running (%s,%d)' % (args.ip, args.port))
    server.start()
    server.waitForThread()
