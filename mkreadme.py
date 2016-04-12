from string import Template
import subprocess
tpl = Template(open("README.tpl.md").read())

q = subprocess.check_output(['python', 'preprocess.py', '-h'])
i = q.find("exit")
q = q[i+5:]

preprocess = []
for l in q.split("--")[1:]:
    t = l.split()
    preprocess.append( "* `%s`: %s"%(t[0], " ".join(t[2:])))


q = subprocess.check_output(['th', 'train.lua', '-h'])
train = []
for l in q.split("\n")[1:]:
    if not l.startswith("  -"):
        train.append(l.strip())
        continue
    t = l.split()
    train.append( "* `%s`: %s"%(t[0], " ".join(t[1:])))

q = subprocess.check_output(['th', 'beam.lua', '-h'])
beam = []
for l in q.split("\n")[1:]:
    if not l.startswith("  -"):
        train.append(l.strip())
        continue
    t = l.split()
    train.append( "* `%s`: %s"%(t[0], " ".join(t[1:])))


print(tpl.substitute({"preprocessargs": "\n".join(preprocess),
                      "trainargs": "\n".join(train),
                      "beamargs": "\n".join(beam) }))
