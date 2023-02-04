import os


def update(*args, user=False, force=False, **kwargs):
    url = "gitlab.com/bigd4/magus.git"
    pip_cmd = "pip install --upgrade"
    if user:
        pip_cmd += " --user"
    if force:
        pip_cmd += " --force-reinstall --no-dependencies"
    print("Updating...")
    print("Try to install by ssh...")
    cmd = pip_cmd + " git+ssh://git@" + url
    print(cmd)
    if os.system(cmd) == 0:
        return True
    else:
        print("SSh update Failed!\n"
              "This may caused by no ssh key in git.nju.edu.cn.\n"
              "Please add your ssh key in the settings")
    print("Try to install by http...")
    cmd = pip_cmd + " git+https://" + url
    print(cmd)
    if os.system(cmd) == 0:
        return True
    else:
        print("Update Failed!")
