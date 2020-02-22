import subprocess
import sys,os
import time
import logging
import datetime

class JobManager:
    def __init__(self):
        self.jobs=[]
        self.history=[]

    def bsub(self,command):
        job=dict()
        jobid=subprocess.check_output(command, shell=True).split()[1][1: -1]
        if type(jobid) is bytes:
            jobid=jobid.decode()
        job['id']=jobid
        job['workDir']=os.getcwd()
        job['subtime']=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.jobs.append(job)

    def checkjobs(self):
        logging.debug("Checking jobs...")
        logging.debug(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        for job in self.jobs:
            try:
                stat = subprocess.check_output("bjobs %s | grep %s | awk '{print $3}'"% (job['id'], job['id']), shell=True)
                stat = stat.decode()[:-1]
            except:
                s = sys.exc_info()
                logging.info("Error '%s' happened on line %d" % (s[1],s[2].tb_lineno))
                stat = ''
            # logging.debug(job['id'], stat)
            if stat == 'DONE' or stat == '':
                job['state'] = 'DONE'
            elif stat == 'PEND':
                job['state'] = 'PEND'
                return False
            elif stat == 'RUN':
                job['state'] = 'RUN'
                return False
            else:
                job['state'] = 'ERROR'
        return True

    def WaitJobsDone(self,waitTime):
        while not self.checkjobs():
            time.sleep(waitTime)

    def clear(self):
        self.history.extend(self.jobs)
        self.jobs=[]