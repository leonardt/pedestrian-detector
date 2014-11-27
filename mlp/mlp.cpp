#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <time.h>
#include <vector>
#include <iterator>
int forward_prop(float *input, int input_size, float *weights, int *dim, int dim_size){
    /* @param input: array of length 36 representing 6 6x1 vectors
     * @param input_size: size of input array.
     * @param weights: weight vector
     * @param dim: array of layer sizes.  dim[0] is size of input, dim[1] is size of 1st hidden layer.   
     * 	       	   dim[dim_size-1] is 1 for our binary classifier.
     * @param sim_size: size of dim array
     */
    int dim_index = 0;
    int weights_index = 0;
    for(dim_index; dim_index < dim_size-2; dim_index++){
	int a = dim[dim_index];
	int b = dim[dim_index+1];
	float layer_weights[a][b]; //weight matrix for this layer
	memcpy(layer_weights, weights, 36*36*sizeof(float));
        //for(int i = 0; i<36; i++){
	//    for(int j=0; j<36; j++){
	//	printf("%2f ", layer_weights[i][j]);
	//    };
	//};
//	std::copy(std::begin(weights), std::begin(weights+a*b), std::begin(layer_weights));
//	printf("~~~~~~~~~~~~~~~~~~~~~\n");
	for(int i=0; i<a; i++){
	    for(int j=0; j<b; j++){
//		printf("%f ", layer_weights[i][j]);
	    }
	}
    };
};






int init(){
    // create weights
    //float weights[36][36] = {};
    float* weights = (float*) malloc(36*36*sizeof(float));
    srand (time(NULL));
    for(int i=0; i<36; i++){
	for(int j=0; j<36; j++){
	    weights[i*36+j] = float(float(rand()%10)/float(10));
	}
    }
    FILE *pFile = fopen("weights", "wb");
    if (!pFile) {
	std::cout<<"error opening file";
	exit(0);
    };
    fwrite(weights, sizeof(float), 36*36, pFile);
    fclose(pFile); 
};

int main(int argc, char* argv[]){
    // initialize weights if it wasn't already 
    // init(...)
    //FILE *pFile = fopen("weights", "rb");
    //if (!pFile) {
//	std::cout<<"error opening file";
//	exit(0);
 //   };
    //float weights[36][36];
    float* weights = (float*) malloc(36*36*sizeof(float));
    srand (time(NULL));
    for(int i=0; i<36; i++){
	for(int j=0; j<36; j++){
	    weights[i*36+j] = float(float(rand()%10)/float(10));
	}
    }

    //fread(weights, sizeof(float), 36*36, pFile);
    //fclose(pFile);
       
    int dim[3] = {36, 36, 1};
    float input[36];
    forward_prop(input, 36, weights, dim, 3);
//    for(int i = 0; i<36*36; i++){
//	printf("%f ", weights[i]);
 //   };
    //{int forward_prop(float *input, int input_size, float *weights, int *dim, int dim_size){
};
