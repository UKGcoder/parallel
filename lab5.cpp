#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <pthread.h>

#define PROC_TASKS 100
#define LISTS 4
#define L 50000
struct info{
    int current_task;
    int remaining_tasks;
    int *tasks;
    pthread_mutex_t p_mutex;
    pthread_t recReceiver;
    pthread_attr_t pthread_attr;
    double result;
    int tasks_number;

};

void structInit(info &args){
    args.current_task=0;
    args.remaining_tasks=PROC_TASKS;
    args.tasks = (int*)calloc(PROC_TASKS, sizeof(int));
    args.result = 0;
    args.tasks_number = 0;
}

static void* receive(void *arg) {
  MPI_Status status;
    struct info* args;
    args = (info*)arg;
  int buffer;
  int prepared_tasks = 0;
  while (1) {
    MPI_Recv(&buffer, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
    if (buffer == -1) break;
    pthread_mutex_lock(&args->p_mutex);
    if (args->remaining_tasks > (PROC_TASKS / 20)) {
      prepared_tasks = args->remaining_tasks / 2;
      MPI_Send(&prepared_tasks, 1, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD);
      MPI_Send(&args->tasks[args->current_task + 1], prepared_tasks, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD);
      args->current_task += prepared_tasks;
      args->remaining_tasks -= prepared_tasks;
    } else {
      prepared_tasks = 0;
      MPI_Send(&prepared_tasks, 1, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD);
    }
    pthread_mutex_unlock(&args->p_mutex);
  }
    pthread_exit(0);
    return NULL;
}

void pthreadInitialize(info &args) {
  pthread_attr_init(&args.pthread_attr);
  pthread_attr_setdetachstate(&args.pthread_attr, PTHREAD_CREATE_JOINABLE);
  pthread_create(&args.recReceiver, &args.pthread_attr, &receive, &args);
  pthread_attr_destroy(&args.pthread_attr);
  pthread_mutex_init(&args.p_mutex, NULL);
}

void doTask(info &args,int itTask){
    for (int j = 0; j < args.tasks[itTask]; j++) {
      args.result += sin(j);
    }
    args.tasks_number++;
}

void calc(info &args) {
  int itTask;
  while (args.remaining_tasks != 0) {
    itTask = args.current_task;
    args.remaining_tasks--;
    pthread_mutex_unlock(&args.p_mutex);
      doTask(args,itTask);
    pthread_mutex_lock(&args.p_mutex);
    args.current_task++;
  }
}

int main(int argc, char **argv) {
    info args;
    structInit(args);
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, NULL);
  int size;
  int rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  pthreadInitialize(args);

  for (int i = 0; i < LISTS; i++) {
    double start = MPI_Wtime();
    for (int j = 0; j < PROC_TASKS; j++) {
      args.tasks[j] = abs(50 - j % 100) * abs(rank - (i % size)) * L;
    }
    pthread_mutex_lock(&args.p_mutex);
    calc(args);
    pthread_mutex_unlock(&args.p_mutex);
    int extra_tasks_flag = 0;
    while (extra_tasks_flag == 0) {
      extra_tasks_flag = 1;
      int itRank = (rank + 1) % size;
      int addTasks;
      while (itRank != rank) {
        MPI_Send(&rank, 1, MPI_INT, itRank, 0, MPI_COMM_WORLD);
        MPI_Recv(&addTasks, 1, MPI_INT, itRank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (addTasks != 0) {
          MPI_Recv(args.tasks, addTasks, MPI_INT, itRank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          pthread_mutex_lock(&args.p_mutex);
          args.current_task = 0;
          args.remaining_tasks = addTasks;
          calc(args);
          pthread_mutex_unlock(&args.p_mutex);
          extra_tasks_flag = 0;
        }
        itRank++;
        itRank %= size;
      }
    }
     
    double end = MPI_Wtime();
    double time = end - start;
    double minTime;
    double maxTime;
    MPI_Allreduce(&time, &minTime, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&time, &maxTime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    printf("[tasks_list %d]\n", i);
    printf("[rank %d]\n tasks_number = %d\n result = %f\n time: %f\n", rank, args.tasks_number, args.result, time);
    MPI_Barrier(MPI_COMM_WORLD);
  }
  int receiver_end = -1;
  MPI_Send(&receiver_end, 1, MPI_INT, (rank + 1) % size, 0, MPI_COMM_WORLD);
  pthread_join(args.recReceiver, NULL);
  pthread_mutex_destroy(&args.p_mutex);
  free(args.tasks);
  MPI_Finalize();
  return 0;
}


