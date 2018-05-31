def input_handler(sequence_path,psiblast_path,db_path,temp_path,n_threads=1,n_iterations=3):
    '''Function that creates sequence profile from an input in FASTA format
    sequence_path: path where the FASTA file lies
    psiblast_path: path where psiblast.exe lies
    db_path: path where the batabase of reference for the alignments lies
    n_threads: number of CPUs to be used
    n_iterations: how many rounds of allignement should psiblast perform for each sequence
    temp_path: path where the produced files will lie'''
    import os
    import numpy as np
    import pandas as pd
    from Bio import SeqIO
    n_iterations=str(n_iterations)
    n_threads=str(n_threads)
    ids=[]
    lengths=[]
    sequences=[]

    #FASTA parser
    for seq in SeqIO.parse(sequence_path, 'fasta'):
        id = f'{seq.id}.faa'
        ids.append(seq.id)
        lengths.append(seq.__len__())
        SeqIO.write(seq, f'{temp_path}\{id}', 'fasta')

    for i, identifier in enumerate(ids):
        path_sequence = f"{temp_path}\{identifier}.faa"
        path_outall = f"{temp_path}\{identifier}.all"
        path_outckp = f"{temp_path}\{identifier}.ckp"
        path_outmat = f"{temp_path}\{identifier}.mat"
        os.system(
            f'{psiblast_path} -query {path_sequence} -db {db_path} -out {path_outall} -out_pssm {path_outckp} -out_ascii_pssm {path_outmat} -num_iterations {n_iterations} -num_threads {n_threads}')
        df = pd.read_table(path_outmat, skiprows=range(0, 2), nrows=lengths[i], sep='\s+')
        act_table = df.iloc[:, 18:-2]
        in_tab = np.array((act_table / 100), dtype='f')
        sequences.append(in_tab)

