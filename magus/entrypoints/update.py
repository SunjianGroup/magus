import os
import unittest

from setuptools import Command
import magus
from BeautifulReport import BeautifulReport


def update(*args, user=True, force=True, **kwargs):
    url = "git.nju.edu.cn/gaaooh/magus.git@devel"
    cmd = "pip install --upgrade"
    if user:
        cmd += " --user"
    if force:
        cmd += " --force-reinstall"
    print("Updating...")
    print("Try to install by ssh...")
    if os.system(cmd + " git+ssh://git@" + url) == 0:
        return True
    else:
        print("SSh update Failed!\n"
              "This may caused by no ssh key in git.nju.edu.cn.\n"
              "Please add your ssh key in the settings")
    print("Try to install by http...")
    if os.system(cmd + " git+https://" + url) == 0:
        return True
    else:
        print("Update Failed!")
