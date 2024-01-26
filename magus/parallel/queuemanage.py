import subprocess, sys, os, time, logging, datetime, yaml
from magus.utils import check_parameters
import re


log = logging.getLogger(__name__)


class BaseJobManager:
    control_keys = ['queue_name', 'num_core', 'pre_processing', 'verbose', 'kill_time', 'mem_per_cpu']
    def __init__(self, **parameters):
        Requirement = ['queue_name', 'num_core']
        Default={
            'control_file': None,
            'pre_processing': 200,
            'verbose': False,
            'kill_time': 7200,
            'mem_per_cpu': '1G'
            }
        check_parameters(self, parameters, Requirement, Default)
        self.jobs = []
        self.history = []
        if self.control_file:
            with open(self.control_file, 'w') as f:
                f.write(yaml.dump({key: getattr(self, key) for key in self.control_keys}))

    def reload(self):
        if self.control_file is not None:
            changed_info = []
            with open(self.control_file) as f:
                control_dict = yaml.load(f, Loader=yaml.FullLoader)
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


# class BSUBSystemManager(BaseJobManager):
#     def kill(self, jobid):
#         subprocess.call('bkill {}'.format(jobid), shell=True)

#     def sub(self, content, name='job', file='job', out='out', err='err'):
#         self.reload()
#         if os.path.exists('DONE'):
#             os.remove('DONE')
#         if os.path.exists('ERROR'):
#             os.remove('ERROR')
#         with open(file, 'w') as f:
#             f.write(
#                 "#BSUB -q {0}\n"
#                 "#BSUB -n {1}\n"
#                 "#BSUB -o {2}\n"
#                 "#BSUB -e {3}\n"
#                 "#BSUB -J {4}\n"
#                 #"#BSUB -R affinity[core:cpubind=core:membind=localprefer:distribute=pack]"
#                 "{5}\n"
#                 "{6}\n"
#                 "[[ $? -eq 0 ]] && touch DONE || touch ERROR".format(self.queue_name, self.num_core, out, err, name, self.pre_processing, content)
#                 )
#         command = 'bsub < ' + file
#         job = dict()
#         jobid = subprocess.check_output(command, shell=True).split()[1][1: -1]
#         if type(jobid) is bytes:
#             jobid = jobid.decode()
#         job['id'] = jobid
#         job['workDir'] = os.getcwd()
#         job['subtime'] = datetime.datetime.now()
#         job['name'] = name
#         job['err'] = err
#         job['out'] = out
#         self.jobs.append(job)
#         return job

#     def check_jobs(self):
#         log.debug("Checking jobs...")
#         nowtime = datetime.datetime.now()
#         log.debug(nowtime.strftime('%m-%d %H:%M:%S'))
#         allDone = True
#         # joblist = subprocess.check_output("bjobs -a", shell=True).decode().split('\n')[1: -1]
#         # time.sleep(10)
#         # jobdict = {job.split()[0]: job.split()[2] for job in joblist}
#         for job in self.jobs:
#             """
#             if job['id'] in jobdict:
#                 stat = jobdict[job['id']]
#             else:
#                 try:
#                     stat = subprocess.check_output("bjobs %s | grep %s | awk '{print $3}'"% (job['id'], job['id']), shell=True)
#                     stat = stat.decode()[:-1]
#                     time.sleep(10)
#                 except:
#                     s = sys.exc_info()
#                     log.warning("Error '%s' happened on line %d" % (s[1],s[2].tb_lineno))
#                     stat = ''
#             """
#             try:
#                 ret = subprocess.check_output("bjobs -noheader -o stat {}".format(job['id']), shell=True).decode().split('\n')[0]
#                 if 'is not found' in ret:
#                     stat = 'NotFound'
#                 else:
#                     stat = ret
#                 time.sleep(1)
#             except:
#                 log.warning("Check Job {} Error".format(job['id']))
#                 stat = ''
#             # log.debug(job['id'], stat)
#             if stat == 'NotFound':
#                 if os.path.exists(os.path.join(job['workDir'], 'DONE')):
#                     job['state'] = 'DONE'
#                 elif os.path.exists(os.path.join(job['workDir'], 'ERROR')):
#                     job['state'] = 'ERROR'
#             if stat == 'DONE' or stat == '':
#                 job['state'] = 'DONE'
#             elif stat == 'PEND':
#                 job['state'] = 'PEND'
#                 allDone = False
#             elif stat == 'SSUSP':
#                 job['state'] = 'SSUSP'
#                 allDone = False
#             elif stat == 'RUN':
#                 if 'begintime' not in job.keys():
#                     job['begintime'] = datetime.datetime.now()
#                 job['state'] = 'RUN'
#                 allDone = False
#                 runtime = (nowtime - job['begintime']).total_seconds()
#                 if runtime > self.kill_time:
#                     self.kill(job['id'])
#                     log.warning('job {} id {} has run {}s, ni pao ni ma ne?'.format(job['name'],job['id'],runtime))
#             else:
#                 job['state'] = 'ERROR'
#             if self.verbose:
#                 log.debug('job {} id {} : {}'.format(job['name'], job['id'], job['state']))
#         return allDone


class LSFSystemManager(BaseJobManager):
    def kill(self, jobid):
        subprocess.call('bkill {}'.format(jobid), shell=True)

    def sub(self, content, name='job', file='job', out='out', err='err'):
        self.reload()
        if os.path.exists('DONE'):
            os.remove('DONE')
        if os.path.exists('ERROR'):
            os.remove('ERROR')
        with open(file, 'w') as f:
            hours = self.kill_time // 3600
            minites = (self.kill_time % 3600) // 60
            f.write(f"#BSUB -q {self.queue_name}\n"
                    f"#BSUB -n {self.num_core}\n"
                    f"#BSUB -o {out}\n"
                    f"#BSUB -e {err}\n"
                    f"#BSUB -J {name}\n"
                    f"#BSUB -W {hours}:{minites}\n"
                    f"{self.pre_processing}\n"
                    f"{content}\n"
                    "[[ $? -eq 0 ]] && touch DONE || touch ERROR"
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
        job['err'] = err
        job['out'] = out
        self.jobs.append(job)
        return job

    def wait_jobs_done(self, wait_time):
        wait_condition = " && ".join(["ended({})".format(job['id']) for job in self.jobs])
        os.system("bwait -w '{}'".format(wait_condition))


class SLURMSystemManager(BaseJobManager):
    def kill(self, jobid):
        subprocess.call('scancel {}'.format(jobid), shell=True)

    def sub(self, content, name='job', file='job', out='out', err='err'):
        self.reload()
        with open(file, 'w') as f:
            hours = self.kill_time // 3600
            minites = (self.kill_time % 3600) // 60
            seconds = int(self.kill_time % 60)
            # In some slurm system, --mem-per-cpu option does not exist, so we manually multiply mem_by_cpu by num_core.
            memory = str(int(re.findall("^\d+", self.mem_per_cpu)[0]) * self.num_core) \
                     + (re.findall("[K|M|G|T]$", self.mem_per_cpu) + [''])[0]
            f.write(
                f"#!/bin/bash\n"
                f"#SBATCH --partition={self.queue_name}\n"
                f"#SBATCH --no-requeue\n"
                f"#SBATCH --mem={memory}\n" # --mem-per-cpu doesn't work for some SLURM systems
                f'#SBATCH --time={hours}:{minites}:{seconds}\n'
                f"#SBATCH --nodes=1\n"
                f"#SBATCH --ntasks-per-node={self.num_core}\n"
                f"#SBATCH --job-name={name}\n"
                f"#SBATCH --output={out}\n"
                f"{self.pre_processing}\n"
                f"{content}\n")
                #.format(self.queue_name, self.num_core, out, err, name, self.pre_processing, content,
                #             time.strftime("%H:%M:%S", time.gmtime(self.kill_time)))


        command = 'sbatch ' + file
        job = dict()
        for _ in range(5):
            try:
                jobid = subprocess.check_output(command, shell=True).split()[-1]
                break
            except:
                pass
        else:
            log.info("Fail to submit job! Error in 'sbatch' command.")
            return None
        if type(jobid) is bytes:
            jobid = jobid.decode()
        job['id'] = jobid
        job['workDir'] = os.getcwd()
        job['subtime'] = datetime.datetime.now()
        job['name'] = name
        self.jobs.append(job)
        # wait a moment so that we can find the job which is just submitted
        time.sleep(3)
        return job

    def check_jobs(self):
        log.debug("Checking jobs...")
        nowtime = datetime.datetime.now()
        log.debug(nowtime.strftime('%m-%d %H:%M:%S'))
        allDone = True
        time.sleep(4)
        for job in self.jobs:
            try:
                stat = subprocess.check_output("sacct --format=jobid,state | grep '%s ' | awk '{print $2}'"% (job['id']), shell=True)
                stat = stat.decode()[:-1]
            except:
                s = sys.exc_info()
                log.warning("Error '%s' happened on line %d" % (s[1],s[2].tb_lineno))
                stat = ''
            log.debug("{}\t{}".format(job['id'], stat))
            if stat == 'COMPLETED' or stat == '':
                job['state'] = 'DONE'
                allDone = True
            elif stat == 'PENDING':
                job['state'] = 'PEND'
                allDone = False
            elif stat == 'RUNNING':
                job['state'] = 'RUN'
                allDone = False
            else:
                job['state'] = 'ERROR'
                allDone = False
            if self.verbose:
                log.debug('job {} id {} : {}'.format(job['name'], job['id'], job['state']))
        return allDone


class PBSSystemManager(BaseJobManager):
    def kill(self, jobid):
        subprocess.call('qdel {}'.format(jobid), shell=True)

    def sub(self, content, name='job', file='job', out='out', err='err'):
        self.current_directory = os.getcwd()
        self.reload()
        if os.path.exists('DONE'):
            os.remove('DONE')
        if os.path.exists('ERROR'):
            os.remove('ERROR')
        with open(file, 'w') as f:
            f.write(
                "#!/bin/bash\n"
                "#PBS -q {0}\n"
                "#PBS -l nodes=1:ppn={1},walltime={7}\n"
                "#PBS -j oe\n"
                "#PBS -V\n"
                "#PBS -N {4}\n"
                #"cd $PBS_O_WORKDIR\n"
                "cd {8}\n"
                "NP=`cat $PBS_NODEFILE|wc -l`\n"
                "{5}\n"
                "{6}".format(self.queue_name, self.num_core, out, err, name, self.pre_processing, content,
                             time.strftime("%H:%M:%S", time.gmtime(self.kill_time)), self.current_directory)
                )
        command = 'qsub  ' + file
        job = dict()
        jobid = subprocess.check_output(command, shell=True).split()[-1]
        if type(jobid) is bytes:
            jobid = jobid.decode()
        job['id'] = jobid
        job['workDir'] = os.getcwd()
        job['subtime'] = datetime.datetime.now()
        job['name'] = name
        #job['err'] = err
        #job['out'] = out
        self.jobs.append(job)
        time.sleep(3)
        return job

    def check_jobs(self):
        log.debug("Checking jobs...")
        nowtime = datetime.datetime.now()
        log.debug(nowtime.strftime('%m-%d %H:%M:%S'))
        allDone = True
        for job in self.jobs:
            try:
                stat = subprocess.check_output("qstat {0} | grep {0} | awk '{{print $5}}'".format(job['id']), shell=True)
                stat = stat.decode()[:-1]
            except:
                s = sys.exc_info()
                log.warning("Error '%s' happened on line %d" % (s[1],s[2].tb_lineno))
                stat = ''
            log.debug("{}\t{}".format(job['id'], stat))
            if stat == 'C' or stat == '':
                job['state'] = 'DONE'
            elif stat == 'Q':
                job['state'] = 'PEND'
                allDone = False
            elif stat == 'R':
                job['state'] = 'RUN'
                allDone = False
            else:
                job['state'] = 'ERROR'
            if self.verbose:
                log.debug('job {} id {} : {}'.format(job['name'], job['id'], job['state']))
        return allDone


JobManager_dict = {
    'LSF': LSFSystemManager,
    'SLURM': SLURMSystemManager,
    'PBS': PBSSystemManager,
}
job_system = os.getenv('JOB_SYSTEM') or 'LSF'
JobManager = JobManager_dict[job_system]
