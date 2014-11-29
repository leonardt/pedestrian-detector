#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <time.h>
#include <vector>
#include <iterator>
#include <math.h> 
#include <Accelerate/Accelerate.h>
using namespace std;

class Hidden_Layer {
    /* class definition that represents a hidden layer. On construction computes the hidden layer
     *  H(x) = s(b+Wx), where s is the activation function and b is the bias vector
     *  Stores the result in to an output variable. 
     *  n_in should be the #elements in input vector and #rows of layer_weights.
     *  n_out is the number of hidden layers and #cols of layer_weights.
     *  Each column of layer_weights corresponds to edges from input to the ith hidden layer element
     */
    float *input, *layer_weights, *bias, *output;
    int n_in, n_out;
public: 
    Hidden_Layer(float*, float *, float *, int, int);

};
Hidden_Layer::Hidden_Layer(float *input, float *layer_weights, float *bias, int n_in, int n_out) {
    input = input;
    layer_weights = layer_weights;
    bias = bias;
    n_in = n_in;
    n_out = n_out;
    output = (float *) malloc(n_in*sizeof(float));
    float *Wx = (float *) malloc(n_in*sizeof(float)); //Product Wx has dimensions n_in * 1
    cblas_sgemv(CblasRowMajor, CblasNoTrans, n_in, n_out, 1.0f, layer_weights, n_in, input, 1, 1.0f, Wx, 1);
    cblas_saxpy(n_in, 1.0, bias, 1, Wx, 1);
    for (int i = 0; i < n_in; i++) {
        Wx[i] = tanh(Wx[i]);
        printf("%f ", Wx[i]);
    }
    output = Wx;

}

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
    	float *layer_weights = (float *) malloc(a*b*sizeof(float)); //weight matrix for this layer
    	memcpy(layer_weights, weights, 36*36*sizeof(float));
        float *bias = (float *) malloc(36*sizeof(float));
        Hidden_Layer layer(input, layer_weights, bias, 36, 36);
    // Need a matrix-vector multiply here. Layer_weights*input 
    //sgemm("n","n", 36, 1, 36, 1, layer_weights, 36, input, 1, 1, result, 36);

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
    float input[36] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    forward_prop(input, 36, weights, dim, 3);
//    for(int i = 0; i<36*36; i++){
//	printf("%f ", weights[i]);
 //   };
    //{int forward_prop(float *input, int input_size, float *weights, int *dim, int dim_size){
};
