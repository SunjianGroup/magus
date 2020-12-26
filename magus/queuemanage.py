import subprocess
import sys,os
import time
import logging
import datetime

class JobManager:
    def __init__(self,verbose=False,killtime=1000000):
        self.verbose = verbose
        self.jobs=[]
        self.history=[]
        self.killtime = killtime

    def bsub(self,command,name):
        job=dict()
        jobid=subprocess.check_output(command, shell=True).split()[1][1: -1]
        if type(jobid) is bytes:
            jobid=jobid.decode()
        job['id']=jobid
        job['workDir']=os.getcwd()
        job['subtime']=datetime.datetime.now()
        job['name']=name
        self.jobs.append(job)

    def checkjobs(self):
        logging.debug("Checking jobs...")
        nowtime = datetime.datetime.now()
        logging.debug(nowtime.strftime('%m-%d %H:%M:%S'))
        allDone = True
        for job in self.jobs:
            try:
                stat = subprocess.check_output("bjobs %s | grep %s | awk '{print $3}'"% (job['id'], job['id']), shell=True)
                stat = stat.decode()[:-1]
            except:
                s = sys.exc_info()
                logging.warning("Error '%s' happened on line %d" % (s[1],s[2].tb_lineno))
                stat = ''
            # logging.debug(job['id'], stat)
            if stat == 'DONE' or stat == '':
                job['state'] = 'DONE'
            elif stat == 'PEND':
                job['state'] = 'PEND'
                allDone = False
            elif stat == 'SSUSP':
                job['state'] = 'SSUSP'
                allDone = False
            elif stat == 'RUN':
                if 'begintime' not in job.keys():
                    job['begintime'] = datetime.datetime.now()
                job['state'] = 'RUN'
                allDone = False
                runtime = (nowtime - job['begintime']).total_seconds()
                if runtime > self.killtime:
                    self.kill(job['id'])
                    logging.warning('job {} id {} has run {}s, ni pao ni ma ne?'.format(job['name'],job['id'],runtime))
            else:
                job['state'] = 'ERROR'
            if self.verbose:
                logging.debug('job {} id {} : {}'.format(job['name'],job['id'],job['state']))
        return allDone
    #TODO kill job after max waittime
    def WaitJobsDone(self,waitTime):
        while not self.checkjobs():
            time.sleep(waitTime)

    def clear(self):
        self.history.extend(self.jobs)
        self.jobs=[]

    def kill(self, jobid):
        subprocess.call('bkill {}'.format(jobid), shell=True)