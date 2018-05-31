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
        lengths.append(len(seq))
        SeqIO.write(seq, f'{temp_path}\{id}', 'fasta')

    #profiles builder
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
    return sequences

def profile_tailor(profiles, N):
    '''Function to get out of a matrix nx20 a matrix Nx20
    works as a tailor: if the profile is too large with respect of N, the function tightens it,
                       if the profile is too small it inserts patches'''
    import numpy as np
    suited_profiles=[]
    for profile in profiles:
        n=profile.shape[0]
        if n<N:
            for i in range(N-n):
                length=profile.shape[0]
                random_index = np.random.randint(low=round((length - 1) / 3), high=round((length - 1) * 2 / 3))
                profile=np.insert(profile,random_index,np.zeros(20),axis=0)
            suited_profiles.append(profile)
        elif n>N:
            for i in range(n-N):
                length = profile.shape[0]
                random_index = np.random.randint(low=round((length - 1) / 3), high=round((length - 1) * 2 / 3))
                profile=np.delete(profile,random_index,axis=0)
            suited_profiles.append(profile)
        else:
            suited_profiles.append(profile)
    return suited_profiles

def profile_patches (suited_profiles):
    '''Function generating booleian vectors about the position of a patch, if present'''
    patches_collector=[]
    import numpy as np
    for profile in suited_profiles:
        patches=[]
        for array in profile:
            patches.append(not np.any(array))
        patches_collector.append(patches)
    return patches_collector

#seqs=input_handler(sequence_path=r"C:\Users\aless\CNN\PROFILpro_1.1\doc\test.fasta",psiblast_path=r"C:\Users\aless\CNN\tools\bin\psiblast.exe",temp_path=r"C:\Users\aless\CNN\temp",n_iterations=3,n_threads=4,db_path=r"C:\Users\aless\CNN\PROFILpro_1.1\data\uniref50\uniref50")
#print(seqs)
#for seq in seqs:
#    print(seq.shape)
#pt=profile_tailor(seqs,100)
#print(pt)
#for i in pt:
#    print(i.shape)
#pp=profile_patches(pt)
#print(pp)