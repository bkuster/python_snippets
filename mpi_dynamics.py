#!/usr/local/bin python3
"""
Addapted from: https://github.com/jbornschein/mpi4py-examples

Demonstrate the task-pull paradigm for high-throughput computing
using mpi4py. Task pull is an efficient way to perform a large number of
independent tasks when there are more tasks than processors, especially
when the run times vary for each task.

This code is over-commented for instructional purposes.
This example was contributed by Craig Finch (cfinch@ieee.org).
Inspired by http://math.acadiau.ca/ACMMaC/Rmpi/index.html

Addapted for highly dynamic workload testing by Ben Kuster
(ben.kuster@bluewin.ch)
"""

from mpi4py import MPI
import time
import numpy as np

def enum(*sequential, **named):
    """Handy way to fake an enumerated type in Python
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

# Define MPI message tags
tags = enum('READY', 'DONE', 'EXIT', 'START')

# Initializations and preliminaries
comm = MPI.COMM_WORLD   # get MPI communicator object
size = comm.size        # total number of processes
rank = comm.rank        # rank of this process
status = MPI.Status()   # get MPI status object
buf = np.zeros(2).astype(int)

if rank == 0:
    # Master process executes code below
    tasks = range(2*size)    # make list of tasks, arbitrary
    task_index = 0           # set itterator
    num_workers = size - 1   # workers are one less then size, since we need the master
    closed_workers = 0       # count of closed workers
    results =  []            # empty list for evaluation
    print("Master starting with %d workers" % num_workers)

    # start looping until no more workers are working
    while closed_workers < num_workers:
        # save some CPU with this snippet
        if not comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG):
          time.sleep(0.2)

        # recieve the workers call, no 'data' vairable needed, since the buffer is filled
        comm.Recv(buf, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)

        source = status.Get_source() # get worker rank
        tag = status.Get_tag()       # get worker tag
        if tag == tags.READY:
            # Worker is ready, so send it a task
            if task_index < len(tasks):
                comm.send(tasks[task_index], dest=source, tag=tags.START)
                print("Sending task %d to worker %d" % (task_index, source))
                task_index += 1
            else: # no tasks left!
                comm.send(None, dest=source, tag=tags.EXIT)
        elif tag == tags.DONE:
            results.append(list(buf))
            print("Got data from worker %d" % source)
        elif tag == tags.EXIT:
            print("Worker %d exited." % source)
            closed_workers += 1

    print("Master finishing")
    # make some nice output
    results = np.vstack(results)
    count = np.asarray(np.unique(results[:,0], return_counts=True)).T
    head = """
core \t | # tasks \t | slept
------------------------------------"""
    print(head)
    for row in count:
      line = "{0} \t | {1} \t\t | {2}"
      slept = row[0]*row[1]*3
      print(line.format(row[0], row[1], slept))

else:
    # create the slave buffer
    buf = np.empty(1)

    # Worker processes execute code below
    while True: # loop until break, break on EXIT tag
        comm.Send(buf, dest=0, tag=tags.READY) # I'm ready!
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status) # get my task
        tag = status.Get_tag() # get its tag

        # check tags
        if tag == tags.START:
            # Do the work here
            result = np.array([rank, task])   # formulate np.array result
            time.sleep(rank*3)                # take a nap based on rank

            # returning results
            comm.Send(result, dest=0, tag=tags.DONE)
            # back to top!
        elif tag == tags.EXIT:
            # master said no more tasks
            break
    # tell master, i'm no longer in the loop
    comm.Send(buf, dest=0, tag=tags.EXIT)

comm.Barrier()
if rank == 0:
  print('just checking')
