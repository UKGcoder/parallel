#include <mpi.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define REQUEST_TAG 10
#define ANSWER_TAG 20
#define PROC_TASKS 200
#define L 10000

typedef struct Task {
    int repeatNum;
} Task;

typedef Task TaskList[PROC_TASKS];

struct Info{
    int rank, size;
    int iterCounter , iterMax;
    int curTask , listSize ;
    double globalRes ;
    TaskList taskList;
    pthread_mutex_t p_mutex;
    MPI_Datatype MPI_TASK;
};

void initInfo(Info &info){
    info.iterCounter=0;
    info.iterMax=5;
    info.curTask=0;
    info.listSize=0;
}

void getTaskList(Info *info) {
    info->listSize = PROC_TASKS;
    for (int i = info->rank * PROC_TASKS; i < (info->rank + 1) * PROC_TASKS; i++) {
        info->taskList[i % PROC_TASKS].repeatNum = abs(PROC_TASKS / 2 - i % PROC_TASKS) * abs(info->rank - (info->iterCounter % info->size)) * L;
    }
}

int getTaskFrom(int from,Info *info) {
    int flag = 2;
    MPI_Send(&flag, 1, MPI_INT, from, REQUEST_TAG, MPI_COMM_WORLD);
    MPI_Recv(&flag, 1, MPI_INT, from, ANSWER_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (flag == 0) return 0;

    Task recvTask;
    MPI_Recv(&recvTask, 1, info->MPI_TASK, from, ANSWER_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    pthread_mutex_lock(&info->p_mutex);
    info->taskList[info->listSize] = recvTask;
    info->listSize++;
    pthread_mutex_unlock(&info->p_mutex);
    return 1;
}

void doTask(Task task,Info *info) {
    for (int i = 0; i < task.repeatNum; i++) {
        info->globalRes += sin(i);
    }
}

void *taskSenderThread(void *args) {
    struct Info* arg = (Info*)args;
    int flag;
    while (arg->iterCounter < arg->iterMax) {
        MPI_Status status;
        MPI_Recv(&flag, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_TAG, MPI_COMM_WORLD, &status);
        if (flag == 3) break;

        pthread_mutex_lock(&arg->p_mutex);
        if (arg->curTask >= arg->listSize - 1) {
            pthread_mutex_unlock(&arg->p_mutex);
            flag = 0;
            MPI_Send(&flag, 1, MPI_INT, status.MPI_SOURCE, ANSWER_TAG, MPI_COMM_WORLD);
            continue;
        }
        arg->listSize--;
        Task sendTask = arg->taskList[arg->listSize];
        pthread_mutex_unlock(&arg->p_mutex);

        flag = 1;
        MPI_Send(&flag, 1, MPI_INT, status.MPI_SOURCE, ANSWER_TAG, MPI_COMM_WORLD);
        MPI_Send(&sendTask, 1, arg->MPI_TASK, status.MPI_SOURCE, ANSWER_TAG, MPI_COMM_WORLD);
    }
    return NULL;
}

void *taskEvaluatorThread(void *args) {
    struct Info* arg = (Info*)args;
    MPI_Barrier(MPI_COMM_WORLD);
    arg->iterCounter = 0;
    while (arg->iterCounter < arg->iterMax) {
        int tasksDone = 0;
        int hasTasks = 1;

        pthread_mutex_lock(&arg->p_mutex);
        arg->curTask = 0;
        getTaskList(arg);
        pthread_mutex_unlock(&arg->p_mutex);

        double start = MPI_Wtime();
        while (hasTasks) {
            pthread_mutex_lock(&arg->p_mutex);
            if (arg->curTask < arg->listSize) {
                Task task = arg->taskList[arg->curTask];
                pthread_mutex_unlock(&arg->p_mutex);
                doTask(task,arg);
                tasksDone++;
                pthread_mutex_lock(&arg->p_mutex);
                arg->curTask++;
                pthread_mutex_unlock(&arg->p_mutex);
                continue;
            }
            arg->curTask = 0;
            arg->listSize = 0;
            pthread_mutex_unlock(&arg->p_mutex);
            hasTasks = 0;
            for (int i = 0; i < arg->size; i++) {
                if (i == arg->rank) continue;
                if (getTaskFrom(i,arg) == 1) {
                    hasTasks = 1;
                }
            }
        }
        double end=MPI_Wtime();

        double timeTaken = end - start;
        printf("Iteration %d, Process %d - tasks done = %d\n", arg->iterCounter, arg->rank, tasksDone);
        printf("Iteration %d, Process %d - globalRes = %lf\n", arg->iterCounter, arg->rank, arg->globalRes);
        printf("Iteration %d, Process %d - time taken = %lf sec.\n", arg->iterCounter, arg->rank, timeTaken);
        fflush(stdout);

        double minTime, maxTime;
        MPI_Reduce(&timeTaken, &minTime, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&timeTaken, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (arg->rank == 0) {
            printf("Iteration %d, Imbalance time = %lf sec., %lf%\n", arg->iterCounter, maxTime - minTime, (maxTime - minTime) / maxTime * 100);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        arg->iterCounter++;
    }

    int flag = 3;
    MPI_Send(&flag, 1, MPI_INT, arg->rank, REQUEST_TAG, MPI_COMM_WORLD);

    return NULL;
}

void createTaskType(Info &info) {
    const int num = 1;
    int lenOfBlocks[num] = { 1 };
    MPI_Datatype types[num] = { MPI_INT };
    MPI_Aint offsets[num];
    offsets[0] = offsetof(Task, repeatNum);
    MPI_Type_create_struct(num, lenOfBlocks, offsets, types, &info.MPI_TASK);
    MPI_Type_commit(&info.MPI_TASK);
}

int main(int argc, char **argv) {
    Info info;
    initInfo(info);
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE) {
        printf("Couldn't init MPI with MPI_THREAD_MULTIPLE level support\n");
        MPI_Finalize();
        return 0;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &info.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &info.size);
    createTaskType(info);

    pthread_t threads[2];
    pthread_mutex_init(&info.p_mutex, NULL);
    pthread_attr_t attrs;
    pthread_attr_init(&attrs);
    pthread_attr_setdetachstate(&attrs, PTHREAD_CREATE_JOINABLE);
    pthread_create(&threads[0], &attrs, taskSenderThread, &info);
    pthread_create(&threads[1], &attrs, taskEvaluatorThread, &info);
    pthread_attr_destroy(&attrs);
    for (int i = 0; i < 2; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_mutex_destroy(&info.p_mutex);
    MPI_Type_free(&info.MPI_TASK);
    MPI_Finalize();
    return 0;
}
