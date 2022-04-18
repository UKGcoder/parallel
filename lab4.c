#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define X 10
#define Y 10
#define ITER 20

void createLife(int* startSpace, int size,int rank){
    for(int y=0;y<size;y++){
        for(int x=0;x<X;x++){
            if(rank==0){
               if((x==1 && y==2) || (x==2 && y==3) || (x==3 && (y==1 || y==2 || y==3))){
                   startSpace[y*X+x]=0;
                 startSpace[y*X+x]+=1;
               }else{
                   startSpace[y*X+x]=0;
               }
           }else{
            startSpace[y*X+x]=0;
           }
        }
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

int checkLife(int* lifeSpace,int pos,int size){
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

void updateLife(int* lifeSpace1,int* lifespace2,int size,int startPos){
    for(int i=startPos*X;i<X*size;i++){
        if(checkLife(lifeSpace1,i,size)==3 || checkLife(lifeSpace1,i,size)==2){
            lifespace2[i]=1;
        }else{
            lifespace2[i]=0;
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
    
    int partSize = Y/process_count;
    int partsRem = Y%process_count;
    if(process_rank<partsRem){
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
    
    
    int iter = 0;
    while(iter<ITER /*|| iterStop(result,process_count)==1*/){
        MPI_Request reqFirst,reqLast,reqNext,reqPrev,reqAll;
        MPI_Isend(lifeSpacePart+X,X,MPI_INT,(process_rank+1)%process_count,123,MPI_COMM_WORLD, &reqFirst);
        MPI_Isend(lifeSpacePart+(X*(partSize-3)),X,MPI_INT,(process_rank+1)%process_count,123,MPI_COMM_WORLD, &reqLast);
        MPI_Irecv(lifeSpacePart,X,MPI_INT,(process_rank+process_count-1)%process_count,123,MPI_COMM_WORLD,&reqPrev);
        MPI_Irecv(lifeSpacePart+(X*(partSize-2)),X,MPI_INT,(process_rank+process_count-1)%process_count,123,MPI_COMM_WORLD,&reqNext);
        vector[process_rank]=arrayEquals(lifeSpacePart,newLifeSpacePart,partSize);
        MPI_Ialltoall(vector,1,MPI_INT,result,1,MPI_INT,MPI_COMM_WORLD,&reqAll);
        if(iter%2==0){
        updateLife(lifeSpacePart,newLifeSpacePart,partSize-2,2);
        }else{
            updateLife(newLifeSpacePart,lifeSpacePart,partSize-2,2);
        }
        MPI_Wait(&reqFirst,MPI_STATUS_IGNORE);
        MPI_Wait(&reqPrev,MPI_STATUS_IGNORE);
        if(iter%2==0){
            updateLife(lifeSpacePart,newLifeSpacePart,2,1);
        }else{
            updateLife(newLifeSpacePart,lifeSpacePart,2,1);
        }
        MPI_Wait(&reqLast,MPI_STATUS_IGNORE);
        MPI_Wait(&reqNext,MPI_STATUS_IGNORE);
        if(iter%2==0){
            updateLife(lifeSpacePart,newLifeSpacePart,partSize-1,partSize-2);
        }else{
            updateLife(newLifeSpacePart,lifeSpacePart,partSize-1,partSize-2);
        }
        
        MPI_Wait(&reqAll,MPI_STATUS_IGNORE);
    
        iter++;
    }
    int* arrOut = (int*)malloc(X*Y*sizeof(int));
    MPI_Gather (lifeSpacePart, X*partSize, MPI_INT, arrOut,
                X*partSize, MPI_INT, 0, MPI_COMM_WORLD);
    if (process_rank == 0) {
        for(int y=0;y<Y;y++){
            for(int x=0;x<X;x++){
                printf("%d",arrOut[y*X+x]);
            }
            printf("\n");
       }
        end_time = MPI_Wtime();
        printf("time taken - %f sec\n", end_time - start_time);
    }
    
    
    
    free(result);
    free(vector);
    free(newLifeSpacePart);
    free(lifeSpacePart);
    MPI_Finalize();
    return 0;
    
}
