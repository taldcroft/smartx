import os
import cPickle as pickle

import jinja2

template = jinja2.Template(open('index_template.html').read())

casepaths = []
for dirpath, dirnames, filenames in os.walk('reports'):
    if 'aoc.pkl' in filenames:
        casepaths.append(dirpath)

casepaths = sorted(casepaths)
aocs = []
for casepath in casepaths:
    print casepath
    aocfile = os.path.join(casepath, 'aoc.pkl')
    aoc = pickle.load(open(aocfile, 'r'))
    aoc.link = '{}/{}/index.html'.format(aoc.case_id, aoc.subcase_id)
    aocs.append(aoc)


out = template.render(aocs=aocs)
with open('reports/index.html', 'w') as f:
    f.write(out)
