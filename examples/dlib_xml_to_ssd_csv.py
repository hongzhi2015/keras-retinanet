import os
import sys
from bs4 import BeautifulSoup

csv = open(sys.argv[1].replace('.xml', '.csv'), 'w')

with open(sys.argv[1], 'r') as f:
  soup = BeautifulSoup(f, 'xml')
  for img in soup.findAll('image'):
    file = img['file']
    boxes = img.findAll('box')
    if len(boxes) == 0:
      csv.write('%s,,,,,\n' % (os.path.basename(file)))
    else:
      for box in img.findAll('box'):
        xmin = box['left']
        xmax = int(box['left']) + int(box['width'])
        ymin = box['top']
        ymax = int(box['top']) + int(box['height'])
        csv.write('%s,%s,%s,%s,%s,1\n' % (os.path.basename(file), xmin, xmax, ymin, ymax))
