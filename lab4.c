#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define X 16
#define Y 16
#define ITER 4

void createLife(int* startSpace, int size,int rank){
    if(size>=3){
        for(int y=0;y<size+2;y++){
            for(int x=0;x<X;x++){
                if(rank==0){
                   if((x==1 && y==1) || (x==2 && y==2) || (y==3 && (x==0 || x==1 || x==2))){
                     startSpace[y*X+x]=1;
                   }else{
                     startSpace[y*X+x]=0;
                   }
                }else{
                startSpace[y*X+x]=0;
            }
         }
       }
    }else if(size==2){
        for(int y=0;y<size+2;y++){
            for(int x=0;x<X;x++){
                if(rank==0){
                   if((x==1 && y==1) || (x==2 && y==2)){
                     startSpace[y*X+x]=1;
                   }else{
                     startSpace[y*X+x]=0;
                   }
                }else if(rank==1){
                    if(y==1 && (x==0 || x==1 || x==2)){
                        startSpace[y*X+x]=1;
                    }else{
                        startSpace[y*X+x]=0;
                    }
                }else{
                startSpace[y*X+x]=0;
              }
         }
       }
    }else{
        printf("too many processes\n");
        exit(0);
    }
}

int arrayEquals(int* array1,int* array2,int size){
    for(int i=0;i<size*X;i++){
        if(array1[i]!=array2[i]){
            return 0;
        }
    }
    return 1;
}

int checkLife(int* lifeSpace,int pos){
    int lifes=0;
    int x = pos%X;
    if(lifeSpace[pos-X]==1){//top
        lifes++;
    }
    if(lifeSpace[pos+X]==1){//under
        lifes++;
    }
    if(lifeSpace[pos+1]==1){
        lifes++;
    }
    if(lifeSpace[pos-1]==1){
        lifes++;
    }
    if(lifeSpace[pos-X+1]==1){
        lifes++;
    }
    if(lifeSpace[pos-1+X]==1){
        lifes++;
    }
    if(x==0){//first element of string
        if(lifeSpace[pos+X+1]==1){//under right
            lifes++;
        }
        if(lifeSpace[pos-1+2*X]==1){//under left
            lifes++;
        }
    }else if(x==X-1){//last element of string
        if(lifeSpace[pos-1-X]==1){//top left
            lifes++;
        }
        if(lifeSpace[pos-X+1-X]==1){//top right
            lifes++;
        }
    }else{
        if(lifeSpace[pos+1+X]==1){//under right
            lifes++;
        }
        if(lifeSpace[pos-1-X]==1){//top left
            lifes++;
        }
    }
    
    return lifes;
}
void createLife2(int* startSpace, int size,int rank){
    for(int y=0;y<size+2;y++){
        for(int x=0;x<X;x++){
            startSpace[y*X+x]=0;
        }
    }
}


void updateLife(int* lifeSpace1,int* lifespace2,int size,int startPos){
    for(int i=startPos*X;i<X*size;i++){
        if(lifeSpace1[i]==1){
          if(checkLife(lifeSpace1,i)==3){
             lifespace2[i]=1;
          }else if(checkLife(lifeSpace1,i)==2){
              lifespace2[i]=1;
          }else{
             lifespace2[i]=0;
           }
        }else if(lifeSpace1[i]==0){
            if(checkLife(lifeSpace1,i)==3 ){
               lifespace2[i]=1;
            }else{
               lifespace2[i]=0;
             }
        }
    }
}

int iterStop(int* array,int size){
    for(int i=0;i<size;i++){
        if(array[i]==0){
            return 1;
        }
    }
    return 0;
}

void  printArray(int* array){
    for(int y=0;y<Y;y++){
        for(int x=0;x<X;x++){
            printf("%d",array[y*X+x]);
        }
        printf("\n");
   }
    printf("\n");
}

void argsForGatherv(int* revCounts,int* displs,int partSize,int process_count,int remSize){
    int sum=0;
    for(int i=0;i<process_count;i++){
        displs[i]=sum;
        if(i<remSize){
            revCounts[i]=X*partSize+X;
        }else{
            revCounts[i]=partSize*X;
        }
        printf("%d ",revCounts[i]);
        printf("%d ",displs[i]);
        sum+=revCounts[i];
    }
    printf("\n");
}

void algorithm(int* lifeSpacePart,int* newLifeSpacePart,int*vector,int* result,int process_count,int process_rank,int partSize,int iter){
    MPI_Request reqFirst,reqLast,reqNext,reqPrev,reqAll;
    if(iter%2==0){
        MPI_Isend(lifeSpacePart+X,X,MPI_INT,(process_rank+process_count-1)%process_count,123,MPI_COMM_WORLD, &reqFirst);//отправка первой строки пред процессу
        MPI_Isend(lifeSpacePart+X*(partSize),X,MPI_INT,(process_rank+1)%process_count,123,MPI_COMM_WORLD, &reqLast);//отправка последнего
        MPI_Irecv(lifeSpacePart,X,MPI_INT,(process_rank+process_count-1)%process_count,123,MPI_COMM_WORLD,&reqPrev);//прием последней строки прошлого процесса
        MPI_Irecv(lifeSpacePart+X*(partSize+1),X,MPI_INT,(process_rank+1)%process_count,123,MPI_COMM_WORLD,&reqNext);//прием  первой строки следующего
        vector[process_rank]=arrayEquals(lifeSpacePart,newLifeSpacePart,partSize);//5
    }else{
        MPI_Isend(newLifeSpacePart+X,X,MPI_INT,(process_rank+process_count-1)%process_count,123,MPI_COMM_WORLD, &reqFirst);//отправка первой строки пред процессу
        MPI_Isend(newLifeSpacePart+X*(partSize),X,MPI_INT,(process_rank+1)%process_count,123,MPI_COMM_WORLD, &reqLast);//отправка последнего
        MPI_Irecv(newLifeSpacePart,X,MPI_INT,(process_rank+process_count-1)%process_count,123,MPI_COMM_WORLD,&reqPrev);//прием последней строки прошлого процесса
        MPI_Irecv(newLifeSpacePart+X*(partSize+1),X,MPI_INT,(process_rank+1)%process_count,123,MPI_COMM_WORLD,&reqNext);//прием  первой строки следующего
    vector[process_rank]=arrayEquals(lifeSpacePart,newLifeSpacePart,partSize);//5
    }
    MPI_Ialltoall(vector,1,MPI_INT,result,1,MPI_INT,MPI_COMM_WORLD,&reqAll);//6
    if(partSize>2){
        if(iter%2==0){
            updateLife(lifeSpacePart,newLifeSpacePart,partSize,2);//7
        }else{
            updateLife(newLifeSpacePart,lifeSpacePart,partSize,2);
        }
    }
    MPI_Wait(&reqFirst,MPI_STATUS_IGNORE);//8
    MPI_Wait(&reqPrev,MPI_STATUS_IGNORE);//9
    if(iter%2==0){
        updateLife(lifeSpacePart,newLifeSpacePart,2,1);//10
    }else{
        updateLife(newLifeSpacePart,lifeSpacePart,2,1);
    }
    MPI_Wait(&reqLast,MPI_STATUS_IGNORE);//11
    MPI_Wait(&reqNext,MPI_STATUS_IGNORE);//12
    if(iter%2==0){
        updateLife(lifeSpacePart,newLifeSpacePart,partSize+1,partSize);//13
    }else{
        updateLife(newLifeSpacePart,lifeSpacePart,partSize+1,partSize);
    }
    MPI_Wait(&reqAll,MPI_STATUS_IGNORE);//14
}

int main(int argc, char **argv) {
    double start_time;
    double end_time;
    int process_count;
    int process_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    if(process_count>Y){
        printf("Count of processes is more than N\n");
        exit(0);
    }
    int* revCounts = (int*)malloc(process_count*sizeof(int));
    int* displs = (int*)malloc(process_count*sizeof(int));
    int partSize = Y/process_count;
    int remSize = Y%process_count;
    if(process_rank==0){
        argsForGatherv(revCounts,displs,partSize,process_count,remSize);
    }
    if(process_rank<remSize){
        partSize++;
    }
    int* vector = (int*)malloc(process_count*sizeof(int));
    int* result=(int*)malloc(process_count*sizeof(int));
    int* lifeSpacePart = (int*)malloc(X*(partSize+2)*sizeof(int));
    int* newLifeSpacePart = (int*)malloc(X*(partSize+2)*sizeof(int));
    if (process_rank == 0) {
            start_time = MPI_Wtime();
        }
    createLife(lifeSpacePart,partSize,process_rank);
    createLife2(newLifeSpacePart,partSize,process_rank);
    
    int* arrOut = (int*)malloc(X*Y*sizeof(int));
    
    MPI_Gatherv (lifeSpacePart+X, partSize*X, MPI_INT, arrOut,
                revCounts,displs, MPI_INT, 0, MPI_COMM_WORLD);
    if (process_rank == 0) {
        printArray(arrOut);
        printf("==========\n");
        printf("\n");
    }
    int iter = 0;
    while(iter<ITER /*|| iterStop(result,process_count)==1*/){
        if(process_count==1){
            if(iter%2==0){
            updateLife(lifeSpacePart,newLifeSpacePart,Y,0);
            }else{
            updateLife(newLifeSpacePart,lifeSpacePart,Y,0);
            }
        }else{
           algorithm(lifeSpacePart,newLifeSpacePart,vector,result,process_count,process_rank,partSize,iter);
        }
        iter++;
    }
    if(ITER%2==0){
        MPI_Gatherv(lifeSpacePart+X, partSize*X, MPI_INT, arrOut,
                    revCounts,displs, MPI_INT, 0, MPI_COMM_WORLD);
    }else{
        MPI_Gatherv(newLifeSpacePart+X, partSize*X, MPI_INT, arrOut,
                    revCounts,displs, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if(process_rank==0){
        end_time=MPI_Wtime();
        printArray(arrOut);
        printf("\n");
        printf("time taken - %f sec\n", end_time - start_time);
    }
    free(revCounts);
    free(arrOut);
    free(result);
    free(vector);
    free(newLifeSpacePart);
    free(lifeSpacePart);
    MPI_Finalize();
    return 0;
    
}
