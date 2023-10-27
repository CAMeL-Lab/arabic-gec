#!/usr/bin/python
from sys import argv,exit,stderr
f=open(argv[1],'r')
for l in f:
	print ' '.join(l.split()[1:])
