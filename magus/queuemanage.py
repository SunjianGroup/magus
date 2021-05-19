import subprocess, sys, os, time, logging, datetime, yaml


log = logging.getLogger(__name__)


class BaseJobManager:
    control_keys = ['queue_name', 'num_core', 'pre_processing', 'verbose', 'kill_time']
    def __init__(self, queue_name, num_core, control_file=None,
                 pre_processing='', verbose=False, kill_time=1000000):
        self.queue_name = queue_name
        self.num_core = num_core
        self.pre_processing = pre_processing
        self.verbose = verbose
        self.jobs = []
        self.history = []
        self.kill_time = kill_time
        self.control_file = control_file
        if control_file is not None:
            with open(control_file, 'w') as f:
                f.write(yaml.dump({key: getattr(self, key) for key in self.control_keys}))

    def reload(self):
        if self.control_file is not None:
            changed_info = []
            with open(self.control_file) as f:
                control_dict = yaml.load(f)
            for key in self.control_keys:
                if key in control_dict:
                    if getattr(self, key) != control_dict[key]:
                        changed_info.append("\t{}: {} -> {}".format(key, getattr(self, key), control_dict[key]))
                        setattr(self, key, control_dict[key])
            if len(changed_info) > 0:
                log.info('Be careful, the following settings are changed')
                for info in changed_info:
                    log.info(info)

    def sub(self, content, *arg, **kwargs):
        raise NotImplementedError

    def kill(self, jobid):
        raise NotImplementedError

    def wait_jobs_done(self, wait_time):
        while not self.check_jobs():
            time.sleep(wait_time)

    def clear(self):
        self.history.extend(self.jobs)
        self.jobs=[]


class BSUBSystemManager(BaseJobManager):
    def kill(self, jobid):
        subprocess.call('bkill {}'.format(jobid), shell=True)

    def sub(self, content, name='job', file='job', out='out', err='err'):
        self.reload()
        with open(file, 'w') as f:
            f.write(
                "#BSUB -q {0}\n"
                "#BSUB -n {1}\n"
                "#BSUB -o {2}\n"
                "#BSUB -e {3}\n"
                "#BSUB -J {4}\n"
                #"#BSUB -R affinity[core:cpubind=core:membind=localprefer:distribute=pack]"
                "{5}\n"
                "{6}".format(self.queue_name, self.num_core, out, err, name, self.pre_processing, content)
                )
        command = 'bsub < ' + file
        job = dict()
        jobid = subprocess.check_output(command, shell=True).split()[1][1: -1]
        if type(jobid) is bytes:
            jobid = jobid.decode()
        job['id'] = jobid
        job['workDir'] = os.getcwd()
        job['subtime'] = datetime.datetime.now()
        job['name'] = name
        self.jobs.append(job)

    def check_jobs(self):
        log.debug("Checking jobs...")
        nowtime = datetime.datetime.now()
        log.debug(nowtime.strftime('%m-%d %H:%M:%S'))
        allDone = True
        for job in self.jobs:
            try:
                stat = subprocess.check_output("bjobs %s | grep %s | awk '{print $3}'"% (job['id'], job['id']), shell=True)
                stat = stat.decode()[:-1]
            except:
                s = sys.exc_info()
                log.warning("Error '%s' happened on line %d" % (s[1],s[2].tb_lineno))
                stat = ''
            # log.debug(job['id'], stat)
            if stat == 'DONE' or stat == '':
                job['state'] = 'DONE'
            elif stat == 'PEND':
                job['state'] = 'PEND'
                allDone = False
            elif stat == 'RUN':
                if 'begintime' not in job.keys():
                    job['begintime'] = datetime.datetime.now()
                job['state'] = 'RUN'
                allDone = False
                runtime = (nowtime - job['begintime']).total_seconds()
                if runtime > self.kill_time:
                    self.kill(job['id'])
                    log.warning('job {} id {} has run {}s, ni pao ni ma ne?'.format(job['name'],job['id'],runtime))
            else:
                job['state'] = 'ERROR'
            if self.verbose:
                log.debug('job {} id {} : {}'.format(job['name'], job['id'], job['state']))
        return allDone


class SLURMSystemManager(BaseJobManager):
    def kill(self, jobid):
        subprocess.call('scancel {}'.format(jobid), shell=True)

    def sub(self, content, name='job', file='job', out='out', err='err'):
        self.reload()
        with open(file, 'w') as f:
            f.write(
                "SBATCH --nodes=1\n"
                "SBATCH --ntasks-per-node={1}\n"
                "SBATCH --time={7}\n"
                "SBATCH --job-name={4}\n"
                "SBATCH --output={2}\n"
                "{5}\n"
                "{6}".format(self.queue_name, self.num_core, out, err, name, self.pre_processing, content)
                )
        command = 'sbatch' + file
        job = dict()
        jobid = subprocess.check_output(command, shell=True).split()[1][1: -1]
        if type(jobid) is bytes:
            jobid = jobid.decode()
        job['id'] = jobid
        job['workDir'] = os.getcwd()
        job['subtime'] = datetime.datetime.now()
        job['name'] = name
        self.jobs.append(job)

    def check_jobs(self):
        log.debug("Checking jobs...")
        nowtime = datetime.datetime.now()
        log.debug(nowtime.strftime('%m-%d %H:%M:%S'))
        allDone = True
        for job in self.jobs:
            try:
                stat = subprocess.check_output("sacct %s | grep %s | awk '{print $3}'"% (job['id'], job['id']), shell=True)
                stat = stat.decode()[:-1]
            except:
                s = sys.exc_info()
                log.warning("Error '%s' happened on line %d" % (s[1],s[2].tb_lineno))
                stat = ''
            # log.debug(job['id'], stat)
            if stat == 'DONE' or stat == '':
                job['state'] = 'DONE'
            elif stat == 'PEND':
                job['state'] = 'PEND'
                allDone = False
            elif stat == 'RUN':
                if 'begintime' not in job.keys():
                    job['begintime'] = datetime.datetime.now()
                job['state'] = 'RUN'
                allDone = False
                runtime = (nowtime - job['begintime']).total_seconds()
                if runtime > self.kill_time:
                    self.kill(job['id'])
                    log.warning('job {} id {} has run {}s, ni pao ni ma ne?'.format(job['name'],job['id'],runtime))
            else:
                job['state'] = 'ERROR'
            if self.verbose:
                log.debug('job {} id {} : {}'.format(job['name'], job['id'], job['state']))
        return allDone


JobManager_dict = {
    'BSUB': BSUBSystemManager,
    'SLURM': SLURMSystemManager,
}
job_system = os.getenv('JOB_SYSTEM') or 'BSUB'
JobManager = JobManager_dict[job_system]
