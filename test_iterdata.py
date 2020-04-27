import torch
from itertools import islice
from random import sample,shuffle
import numpy as np 


class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start, end):
         super(MyIterableDataset).__init__()
         assert end > start, "this example code only works with end >= start"
         self.start = start
         self.end = end
         self.data = np.random.randint(low=0,high=5,size=(100,2,2))
         self.dataf = np.random.randn(100)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        return iter(zip(self.data[np.arange(iter_start,iter_end)],self.dataf[np.arange(iter_start,iter_end)]))

ds = MyIterableDataset(start=2, end=10)
dl = list(torch.utils.data.DataLoader(ds, num_workers=0,batch_size=2))

print("-------------------------------------------------------------")
print("All the Data")
print(ds.data.shape)

print(ds.data[2:10])
print(ds.dataf[2:10])

print("-------------------------------------------------------------")

print("Non shuffled list of batch of samples for training ")
print(dl)

print("-------------------------------------------------------------")
### the "sample" function can shuffle the order of the element of the list

print("shuffled list of batch of samples for training ")
print(sample(dl,k=len(dl)))

print("-------------------------------------------------------------")
print("-------------------------------------------------------------")
print("-------------------------------------------------------------")
print("-------------------------------------------------------------")
print("Now testing with AudioToEmbeddingsIterableDataset")
#### test with the new iterable dataset class for movie10


from train_utils import AudioToEmbeddingsIterableDataset
import os


path = '/home/nfarrugi/git/neuromod/cneuromod/movie10/stimuli'

fmripath = '/home/nfarrugi/movie10_parc/sub-01'

for root, dirs, files in os.walk(path, topdown=False):
    for name in files:
        if name[-3:] == 'mkv':
            try:
                currentvid = os.path.join(root, name)
                print(currentvid)
                dataset = AudioToEmbeddingsIterableDataset(currentvid,fmripath=fmripath,samplerate=22050)
                
            except FileNotFoundError as expr:
                print("Issue with file {}".format(currentvid))
                #print(expr)


### quick test with dataloader 

### Here batch size corresponds to the number of successive TR that we take
### We make it a "list" so that we can shuffle its order by using the sample function
### See example above to show how its handled with and without the shuffling
trainloader = list(torch.utils.data.DataLoader(dataset,batch_size=50))

for wav,audioset,imagenet,places,fmri in sample(trainloader,k=len(trainloader)):
    print(wav.shape,audioset.shape,fmri.shape)


print("-------------------------------------------------------------")

#### testing the concatenation of dataloaders using a simple + between lists 
""" trainloader_concat = []
trainloader_concat = trainloader_concat + trainloader
trainloader_concat = trainloader_concat + trainloader

print(len(trainloader_concat))

for wav,audioset,imagenet,places,fmri in sample(trainloader_concat,k=len(trainloader_concat)):
    print(wav.shape,audioset.shape,fmri.shape)
 """


from train_utils import construct_iter_dataloader

trainloader,valloader,testloader = construct_iter_dataloader(path,fmripath,bsize=10)

print(len(trainloader),len(valloader),len(testloader))

for wav,audioset,imagenet,places,fmri in sample(valloader,k=len(valloader)):
    print(wav.shape,audioset.shape,fmri.shape)


#print(wav.shape,audioset.shape,fmri.shape)

print('Success!')