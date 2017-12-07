#-*- coding: utf-8 -*-

import subprocess
import os
import sys

def start_model(path,m):
  global treetagger
  global nbuf
  treetagger = subprocess.Popen([path+'/tree-tagger', m], 
                           stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=open(os.devnull, 'w'),
			   universal_newlines=True)
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
  treetagger.stdin.flush()
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

from http.server import BaseHTTPRequestHandler,HTTPServer
from socketserver import ThreadingMixIn
import threading
import argparse
import re
import cgi

class LocalData(object):
  records = {}

class HTTPRequestHandler(BaseHTTPRequestHandler):
  def do_POST(self):
    if None != re.search('/pos', self.path):
      length = int(self.headers.get('Content-Length'))
      sent=self.rfile.read(length).decode('utf-8')
      self.send_response(200)
      self.end_headers()
      self.wfile.write(tag(sent).encode('utf-8'))
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
