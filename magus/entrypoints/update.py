import os
import unittest
import magus
from BeautifulReport import BeautifulReport


def update(*args, **kwargs):
    print("Updating...")
    print("Try to install by ssh...")
    if os.system("pip install --upgrade --user git+ssh://git@git.nju.edu.cn/gaaooh/magus.git@devel") == 0:
        return True
    else:
        print("SSh update Failed!\n"
              "This may caused by no ssh key in git.nju.edu.cn.\n"
              "Please add your ssh key in the settings")
    print("Try to install by http...")
    if os.system("pip install --upgrade --user git+https://git.nju.edu.cn/gaaooh/magus.git@devel") == 0:
        return True
    else:
        print("Update Failed!")
