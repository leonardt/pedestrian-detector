#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <time.h>
#include <vector>
#include <iterator>
#include <math.h> 
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>  //Apple's lapack 
#else
#include <cblas.h>		    //C lapack
#endif
using namespace std;
class Hidden_Layer; 
void softmax(float *input, float *output, int n);
int forward_prop(float *input, int input_size, Hidden_Layer* hiddenlayers, int numlayers, float* output);
void init(int numlayers, int* layer_sizes, Hidden_Layer* hiddenlayers); 
class Hidden_Layer {
    /* class definition that represents a hidden layer. On construction computes the hidden layer
     *  H(x) = s(b+Wx), where s is the activation function and b is the bias vector
     *  Stores the result in to an output variable. 
     *  n_in should be the #elements in input vector and #rows of layer_weights.
     *  n_out is the number of hidden layers and #cols of layer_weights.
     *  Each column of layer_weights corresponds to edges from input to the ith hidden layer element
     */
 public: 
    float *layer_weights, *bias, *output;
    int n_in, n_out;
    Hidden_Layer(){};
    Hidden_Layer(float *, float *, int, int);
    void compute_output(float* input, int last_layer);

};
Hidden_Layer::Hidden_Layer(float* weights, float* b, int in, int out) {
    layer_weights = weights;
    bias = b;
    n_in = in;
    n_out = out;
    output = (float *) malloc(n_out*sizeof(float));
}
void Hidden_Layer::compute_output(float* input, int last_layer) {
    printf("nout:%d, nin:%d, input[0]: %f", n_out, n_in, input[0]);
    if (n_out != 1) 
        cblas_sgemv(CblasRowMajor, CblasNoTrans, n_out, n_in, 1.0f, layer_weights, n_out, input, 1, 1.0f, output, 1); //computes Wx
    else 
        output[0] = cblas_sdot(n_in, layer_weights, 1, input, 1);
        printf("output is %f \n", output[0]);
    cblas_saxpy(n_out, 1.0, bias, 1, output, 1);
    for (int i = 0; i < n_out; i++) {
        if (!last_layer)
            output[i] = tanh(output[i]);
        else {
            softmax(output, output, n_out);
        }
        printf("%f ", output[i]);
    }
}


int forward_prop(float *input, int input_size, Hidden_Layer* hiddenlayers, int numlayers, float* output) {
    /* @param input: array of length 36 representing 6 6x1 vectors
     * @param input_size: size of input array.
     * @param weights: weight vector
     * @param dim: array of layer sizes.  dim[0] is size of input, dim[1] is size of 1st hidden layer.   
     * 	       	   dim[dim_size-1] is 1 for our binary classifier.
     * @param sim_size: size of dim array
     */
     hiddenlayers[0].compute_output(input, 0);
     for (int i = 1; i < numlayers-2; i++) {
        hiddenlayers[i].compute_output(hiddenlayers[i-1].output, 0);
     }
     hiddenlayers[numlayers-2].compute_output(hiddenlayers[numlayers-3].output, 1);

};


void softmax(float *input, float *output, int n) {
    /*
     * Softmax function computes softmax of given vector.
     *	 out[i] = e^(input[i]) / (sum(k=1 to n){e^(input[k])}) for i in input
     * possible optimization by storing the results of exp() so we only need to call exp() once per input.
     * @param input: the vector to compute on. This would be the intermediate result of the last hidden layer b+Wx.
     * @param output: the vector to store the result.
     * @param n: size of input vector
     */
    float denom = 0.0;
    for (int i=0; i<n; i++){
	   denom += exp(input[i]);
    }
    for(int i=0; i<n; i++){
	   output[i] = exp(input[i]) / denom;
    }
}

void init(int numlayers, int* layer_sizes, Hidden_Layer* hiddenlayers) {
    /* Initializes weights and hidden layers 
     * @param numlayers is the total number of layers. Number of hidden layers is numlayers - 2
     * @param layer_sizes is an int array of size numlayers that stores the sizes of each layer. 
     * weights are initialized randomly or read in from an existing file. 
     */
    int weights_size = 0;
    int bias_size = 0;
    for (int k = 0; k < numlayers-1; k++) {
        weights_size += layer_sizes[k]*layer_sizes[k+1];
        bias_size += layer_sizes[k];
    }
    bias_size -= layer_sizes[numlayers-1];
    float* weights = (float*) malloc(weights_size*sizeof(float));
    float* bias = (float*) malloc(bias_size*sizeof(float));
    srand (time(NULL));
    for (int i=0; i<weights_size; i++){
	    //weights[i] = float(float(rand()%10)/float(10));
        weights[i] = 0.5;
    }
    for (int i = 0; i < bias_size; i++) {
        //bias[x] = float(float(rand()%10)/float(10));
        bias[i] = 0.5;
    }
    //Hidden_Layer* hiddenlayers = new Hidden_Layer[numlayers-1];
    int weights_offset = 0;
    int bias_offset = 0;
    for (int j = 0; j < numlayers-1; j++) {
        Hidden_Layer temp(weights+weights_offset, bias+bias_offset, layer_sizes[j], layer_sizes[j+1]);
        printf("weights: %f ", *(weights+weights_offset));

        printf("temp layer: %f \n", temp.layer_weights[0]);
        hiddenlayers[j] = temp;
        weights_offset += layer_sizes[j] * layer_sizes[j+1];
        bias_offset += layer_sizes[j];
    }



 //    FILE *pFile = fopen("weights", "wb");
 //    if (!pFile) {
	// std::cout<<"error opening file";
	// exit(0);
 //    };
 //    fwrite(weights, sizeof(float), 36*36, pFile);
 //    fclose(pFile); 
};

int main(int argc, char* argv[]){

    Hidden_Layer* hiddenlayers = new Hidden_Layer[2];
    int layer_sizes[3] = {36, 36, 1};
    init(3,layer_sizes, hiddenlayers);
    //fread(weights, sizeof(float), 36*36, pFile);
    //fclose(pFile);
    float output[1];
    float input[36] = {0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01};

    forward_prop(input, 36, hiddenlayers, 3, output);

};
