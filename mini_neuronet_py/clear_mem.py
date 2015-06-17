from os import listdir, remove
from os.path import isfile, join
from sys import argv

for file_name in listdir(argv[1]):
  hp = join(argv[1], file_name)
  for img in listdir(hp):
    remove(join(hp, img))