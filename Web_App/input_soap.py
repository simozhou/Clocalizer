from zeep import Client as cl
import time
import numpy as np
from io import StringIO
import re
import subprocess as sp
import os

global client
client = cl("http://www.ebi.ac.uk/Tools/services/soap/psisearch?wsdl")


def make_request(seq, prev_req=None):
    params = {'sequence': str(seq),
              'database': 'uniref50'}
    if prev_req is not None:
        params['previousjobid'] = prev_req

    jobID = client.service.run(email="simone.procaccia@gmail.com", title="Clocalizer", parameters=params)
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


def psiblaster(seq):
    CWD = os.path.dirname(os.path.realpath(__file__))
    print(CWD)
    psiblast = sp.Popen("./bin/psiblast -db ./bin/uniref50 -out_ascii_pssm temp.mat -num_iterations 3 -num_threads 8",
                        stdin=sp.PIPE,
                        stdout=sp.PIPE, stderr=sp.STDOUT, cwd=CWD, shell=True, executable="/usr/bin/bash")
    alignment, err = psiblast.communicate(bytes(seq, 'utf-8'))

    # with open("temp.fasta", "w") as f:
    #     f.write(seq)
    # command = "psiblast -db uniref50 -query temp.fasta -out_ascii_pssm temp.mat -num_iterations 3 -num_threads 8"
    # alignment = sp.check_output(command, shell=True, executable='/bin/bash')
    #print("!!!!!"+str(alignment))



    with open('temp.mat', 'rb') as f:
        result = f.read()

    #os.system("rm temp.mat")
    #debug result and rm
    sp.Popen("./bin/rm temp.mat",cwd=CWD,shell=True,executable="/usr/bin/bash").communicate()
    #end debug


    coded_results = result.decode('utf8')
    coded_results = re.sub(" +", " ", coded_results)
    coded_results = StringIO(coded_results)

    matrix = np.genfromtxt(coded_results, delimiter=" ", skip_header=3, skip_footer=5, filling_values="??",
                           encoding='utf8', usecols=range(22, 42))

    matrix = np.divide(matrix, 100)

    return matrix, err


def tailor(sequence, n=1000):
    """
    returns a sequence of shape Nx20 by adding zeros at the end or removing elements from the middle
    """
    # if it is only one sequence, we want to convert this 2d array into a tensor to be fed to the model
    if len(sequence.shape) < 3:
        sequence = np.reshape(sequence, (1,) + sequence.shape)

    length = sequence.shape[1]
    difference = length - n

    if difference < 0:
        tailored = np.concatenate((sequence, np.zeros((1, abs(difference), 20), dtype=np.float32)), axis=1)

    elif difference > 0:
        span = range(round((length - 1) / 3), round((length - 1) * 2 / 3) + 1)
        tailored = np.delete(sequence, np.random.choice(span, abs(difference), replace=False, ), axis=1)

    else:
        tailored = sequence

    return tailored


def reshaper(seq_set, model):
    '''
    adds, if not present, the correct nuber of dimensions to fit the model
    :param seq_set:
    :param model:
    :return:
    '''
    if model != "cnn":
        feedable = np.reshape(seq_set, (seq_set.shape[0], 1, seq_set.shape[1], seq_set.shape[2], 1))
    else:
        feedable = np.reshape(seq_set, seq_set.shape + (1,))

    return feedable


def ohe_tailor(ohe, N=1000):
    '''Function to get out of a matrix nx20 a matrix Nx20
    works as a tailor: if the profile is too large with respect of N, the function tightens it,
                       if the profile is too small it inserts patches'''
    n = ohe.shape[0]

    if n < N:

        for i in range(N - n):
            length = ohe.shape[0]
            random_index = np.random.randint(low=round((length - 1) / 3), high=round((length - 1) * 2 / 3))
            ohe = np.insert(ohe, random_index, np.zeros(20, dtype=np.float32), axis=0)
        if len(ohe.shape) < 3:
            ohe = np.reshape(ohe, (1,) + ohe.shape)
        return ohe
    elif n > N:
        for i in range(n - N):
            length = ohe.shape[0]
            random_index = np.random.randint(low=round((length - 1) / 3), high=round((length - 1) * 2 / 3))
            ohe = np.delete(ohe, random_index, axis=0)
        if len(ohe.shape) < 3:
            ohe = np.reshape(ohe, (1,) + ohe.shape)
        return ohe
    else:
        if len(ohe.shape) < 3:
            ohe = np.reshape(ohe, (1,) + ohe.shape)
        return ohe


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
    import time

    start = time.time()
    output = make_input(
        "MYDADEDMQYDEDDDEITPDLWQEACWIVISSYFDEKGLVRQQLDSFDEFIQMSVQRIVEDAPPIDLQAEAQHASGEVEEPPRYLLKFEQIYLSKPTHWERDGAPSPMMPNEARLRNLTYSAPLYVDITKTVIKEGEEQLQTQHQKTFIGKIPIMLRSTYCLLNGLTDRDLCELNECPLDPGGYFIINGSEKVLIAQEKMATNTVYVFAKKDSKYAYTGECRSCLENSSRPTSTIWVSMLARGGQGAKKSAIGQRIVATLPYIKQEVPIIIVFRALGFVSDRDILEHIIYDFEDPEMMEMVKPSLDEAFVIQEQNVALNFIGSRGAKPGVTKEKRIKYAKEVLQKEMLPHVGVSDFCETKKAYFLGYMVHRLLLAALGRRELDDRDHYGNKRLDLAGPLLAFLFRGMFKNLLKEVRIYAQKFIDRGKDFNLELAIKTRIISDGLKYSLATGNWGDQKKAHQARAGVSQVLNRLTFASTLSHLRRLNSPIGRDGKLAKPRQLHNTLWGMVCPAETPEGHAVGLVKNLALMAYISVGSQPSPILEFLEEWSMENLEEISPAAIADATKIFVNGCWVGIHKDPEQLMNTLRKLRRQMDIIVSEVSMIRDIREREIRIYTDAGRICRPLLIVEKQKLLLKKRHIDQLKEREYNNYSWQDLVASGVVEYIDTLEEETVMLAMTPDDLQEKEVAYCSTYTHCEIHPSMILGVCASIIPFPDHNQSPRNTYQSAMGKQAMGVYITNFHVRMDTLAHVLYYPQKPLVTTRSMEYLRFRELPAGINSIVAIASYTGYNQEDSVIMNRSAVDRGFFRSVFYRSYKEQESKKGFDQEEVFEKPTRETCQGMRHAIYDKLDDDGLIAPGVRVSGDDVIIGKTVTLPENEDELESTNRRYTKRDCSTFLRTSETGIVDQVMVTLNQEGYKFCKIRVRSVRIPQIGDKFASRHGQKGTCGIQYRQEDMPFTCEGITPDIIINPHAIPSRMTIGHLIECLQGKVSANKGEIGDATPFNDAVNVQKISNLLSDYGYHLRGNEVLYNGFTGRKITSQIFIGPTYYQRLKHMVDDKIHSRARGPIQILNRQPMEGRSRDGGLRFGEMERDCQIAHGAAQFLRERLFEASDPYQVHVCNLCGIMAIANTRTHTYECRGCRNKTQISLVRMPYACKLLFQELMSMSIAPRMMSV")
    end = (time.time() - start) / 60
    print(output, f'{end} mins total')

    # 15.490066476662953 mins total -- that's a lot

    # import pickle
    #
    # o = pickle.dump(get_results("psisearch-S20180608-140601-0766-67820351-p2m"),
    #                 open("/home/simone/Documents/biochem/dna_pol.pickle", 'wb'))

    a = np.zeros((10, 20))
    print(a)
    print(tailor(a, 6).shape)
    print(tailor(a, 20).shape)
