#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <pthread.h>

#define PROC_TASKS 100
#define LISTS 4
#define L 50000

int current_task = 0;
int remaining_tasks = PROC_TASKS;
int *tasks;
pthread_mutex_t p_mutex;
pthread_t recReceiver;
pthread_attr_t pthread_attr;
double result;
int tasks_number;

void *receive(void *) {
  MPI_Status status;
  int buffer;
  int prepared_tasks = 0;
  while (1) {
    MPI_Recv(&buffer, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
    if (buffer == -1) break;
    pthread_mutex_lock(&p_mutex);
    if (remaining_tasks > (PROC_TASKS / 20)) {
      prepared_tasks = remaining_tasks / 2;
      MPI_Send(&prepared_tasks, 1, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD);
      MPI_Send(&tasks[current_task + 1], prepared_tasks, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD);
      current_task += prepared_tasks;
      remaining_tasks -= prepared_tasks;
    } else {
      prepared_tasks = 0;
      MPI_Send(&prepared_tasks, 1, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD);
    }
    pthread_mutex_unlock(&p_mutex);
  }
  pthread_exit(0);
}

void pthreadInitialize() {
  pthread_attr_init(&pthread_attr);
  pthread_attr_setdetachstate(&pthread_attr, PTHREAD_CREATE_JOINABLE);
  pthread_create(&recReceiver, &pthread_attr, receive, NULL);
  pthread_attr_destroy(&pthread_attr);
  pthread_mutex_init(&p_mutex, NULL);
}

void calc() {
  int itTask;
  while (remaining_tasks != 0) {
    itTask = current_task;
    remaining_tasks--;
    pthread_mutex_unlock(&p_mutex);
    for (int j = 0; j < tasks[itTask]; j++) {
      result += sin(j);
    }
    tasks_number++;
    pthread_mutex_lock(&p_mutex);
    current_task++;
  }
}

int main(int argc, char **argv) {
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, NULL);
  int size;
  int rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  pthreadInitialize();

  tasks = (int*)calloc(PROC_TASKS, sizeof(int));
  result = 0;
  tasks_number = 0;

  for (int i = 0; i < LISTS; i++) {
    double start = MPI_Wtime();
    for (int j = 0; j < PROC_TASKS; j++) {
      tasks[j] = abs(50 - j % 100) * abs(rank - (i % size)) * L;
    }
    pthread_mutex_lock(&p_mutex);
    calc();
    pthread_mutex_unlock(&p_mutex);
    int extra_tasks_flag = 0;
    while (extra_tasks_flag == 0) {
      extra_tasks_flag = 1;
      int itRank = (rank + 1) % size;
      int addTasks;
      while (itRank != rank) {
        MPI_Send(&rank, 1, MPI_INT, itRank, 0, MPI_COMM_WORLD);
        MPI_Recv(&addTasks, 1, MPI_INT, itRank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (addTasks != 0) {
          MPI_Recv(tasks, addTasks, MPI_INT, itRank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          pthread_mutex_lock(&p_mutex);
          current_task = 0;
          remaining_tasks = addTasks;
          calc();
          pthread_mutex_unlock(&p_mutex);
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
    double delta = maxTime - minTime;
    double disbalance = (delta / maxTime) * 100;
    printf("[tasks_list %d]\n", i);
    printf("[rank %d]\n tasks_number = %d\n result = %f\n time: %f\n", rank, tasks_number, result, time);
    printf("delta: %f, disbalance: %f\n", delta, disbalance);
    MPI_Barrier(MPI_COMM_WORLD);
  }
  int receiver_end = -1;
  MPI_Send(&receiver_end, 1, MPI_INT, (rank + 1) % size, 0, MPI_COMM_WORLD);
  pthread_join(recReceiver, NULL);
  pthread_mutex_destroy(&p_mutex);
  MPI_Finalize();
  return 0;
}


