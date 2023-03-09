from streams.utils import FL, FU, generate_stream_section
from  streams.stream_section import StreamSection

def train_and_evaluate(datastream,dataset_name,Q,P):
    '''
    Main evaluation and traing function
    '''
    stream_set = [] #TODO chcek where this should be placed
    for q in Q:
        initial_stream = generate_stream_section(datastream,f'{dataset_name}_init_{q[0]}_{q[1]}',dataset_name,q[0],q[1])
        stream_set.append(initial_stream)
        for p in P:
            ssl_stream = StreamSection(f'{dataset_name}_ssl_{p}_{q[0]}_{q[1]}', FU(initial_stream.stream,p),False)
            lfs_stream = StreamSection(f'{dataset_name}_lfs_{p}_{q[0]}_{q[1]}',FL(ssl_stream.stream),True)
            print(lfs_stream)
            stream_set.append(ssl_stream)
            stream_set.append(lfs_stream)
    