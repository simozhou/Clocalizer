from zeep import Client as cl
import time
from pyspin.spin import make_spin, Default
import numpy as np
from io import StringIO
import re

global client
client = cl("http://www.ebi.ac.uk/Tools/services/soap/psisearch?wsdl")


def make_request(seq, prev_req=None):
    params = {'sequence': str(seq),
              'database': 'uniref50'}
    if prev_req is not None:
        params['previousjobid'] = prev_req

    jobID = client.service.run(email="jan.delshad@gmail.com", title="colocalizer", parameters=params)
    return jobID


def get_status(jobid):
    return client.service.getStatus(jobId=jobid)


def get_results(jobid):
    coded_results = client.service.getResult(jobId=jobid, type="pssm")
    coded_results = coded_results.decode('utf8')
    coded_results = re.sub(" +", " ", coded_results)
    coded_results = StringIO(coded_results)

    matrix = np.genfromtxt(coded_results, delimiter=" ", skip_header=3, skip_footer=5, filling_values="??",
                           encoding='utf8', usecols=range(22, 42))
    matrix = np.divide(matrix, 100)
    return matrix


def make_input(sequence):
    jobID1 = make_request(sequence)
    print(jobID1)
    while get_status(jobID1) == "RUNNING":
        time.sleep(2)
    if get_status(jobID1) == "FINISHED":
        print("First iteration done, second iteration being initiated...")
    else:
        raise RuntimeError(get_status(jobID1))
        return

    jobID2 = make_request(sequence, jobID1)
    print(jobID2)
    while get_status(jobID2) == "RUNNING":
        time.sleep(2)
    if get_status(jobID2) == "FINISHED":
        print("results ready and being processed")
        return get_results(jobID2)
    else:
        raise RuntimeError(get_status(jobID2))
        return


if __name__ == '__main__':
    # output = make_input("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR")
    # print(output)
    # while get_status(output) == "RUNNING":
    #    time.sleep(2)
    # print("finished")
    o = get_results("psisearch-S20180602-173445-0029-93524488-p1m")
    print(o)
