# -*- coding: utf-8 -*-

import subprocess
import os
from http.server import BaseHTTPRequestHandler,HTTPServer
from socketserver import ThreadingMixIn
import threading
import argparse
import re


def start_model(path, m, l, l_first):
  global treetagger
  global nbuf
  global only_lemma
  global lemma_first
  lemma_first = l_first

  only_lemma = False

  tagger_option = ''
  lemma = False
  lemma_no_unknown = ''
  if l == 'none':
      tagger_option = None
  elif l == 'with' or l == 'only':
      if l == 'only':
          only_lemma = True
      tagger_option = '-lemma'
      lemma_no_unknown = '-no-unknown'

  if tagger_option is not None:
      lemma = True

  treetagger = subprocess.Popen([path+'/tree-tagger']+[tagger_option] * lemma + [lemma_no_unknown] * lemma + [m],
                                stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=open(os.devnull, 'w'),
                                universal_newlines=True)

  # this parameter should be adjusted for each os - it forces tree-tagger to flush by following this many sentence ends
  nbuf = 3000


extraneous = 0


def tag(s):
  global extraneous
  joiner_char = '￭'
  l = s.split()
  for w in l:
    # if there's a joiner before the word, get rid of the joiner
    if w[0] == joiner_char:
      treetagger.stdin.write(w[1:]+'\n')
    else:
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
        if only_lemma:
            tag = tag[tag.index('\t')+1:]
        if lemma_first:
            tag = tag[tag.index('\t')+1:]+'\t'+tag[0:tag.index('\t')]
        result.append(tag)
      else:
        extraneous -= 1
      if len(result) == len(l):
        break
  extraneous = nbuf

  return " ".join(result)


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
  parser.add_argument('-lemma', default="none", choices=['none', 'with',' only'], help='Tag with lemmas. \
          \'with\' appends the lemma after the tag, \'only\' appends only the lemma')
  parser.add_argument('-lemma_first', action='store_true', help='when tagging both with pos and lemmas, append the lemma first')
  args = parser.parse_args()

  start_model(args.path, args.model, args.lemma, args.lemma_first)

  if args.sent and len(args.sent):
    print(tag(args.sent))
  else:
    server = SimpleHttpServer(args.ip, args.port)
    print('HTTP Server Running (%s,%d)' % (args.ip, args.port))
    server.start()
    server.waitForThread()
