#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <time.h>
#include <vector>
#include <iterator>
#include <math.h> 
#include <assert.h>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>  //Apple's lapack 
#else
#include <cblas.h>		    //C lapack
#endif
using namespace std;
class Hidden_Layer; 
void softmax(float *input, float *output, int n);
void forward_prop(float *input, int input_size, Hidden_Layer* hiddenlayers, int numlayers);
void init(int numlayers, int* layer_sizes, Hidden_Layer* hiddenlayers); 
float loss(float* input, float* output, int n);

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
    printf("nout:%d, nin:%d \n", n_out, n_in);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, n_out, n_in, 1.0f, layer_weights, n_in, input, 1, 1.0f, output, 1); //computes Wx
    cblas_saxpy(n_out, 1.0, bias, 1, output, 1);
    for (int i = 0; i < n_out; i++) {
        if (!last_layer)
            output[i] = tanh(output[i]);
        else {
            softmax(output, output, n_out);
        }
    }
}


void forward_prop(float *input, int input_size, Hidden_Layer* hiddenlayers, int numlayers) {
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


void backprop(int input, int input_size, Hidden_Layer* hiddenlayers, int numlayers, float* actual) {
    float cost1 = cost(input, input_size, hiddenlayers, numlayers, actual);
    //final error = (a_L - y) __Hadamard__ softmax'(Wx+b), where Hadamard is element wise product
    float* a_L = hiddenlayers[numlayers-2].output ;
    for(int i=0; i<hiddenlayers[numlayers-2].n_out; i++) {
	a_L[i] -= actual[i];
	a_L[i] *= softmax_prime(
    }

}


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

float cost(float* x, int input_size, Hidden_Layer* hiddenlayers, int numlayers, float* y) {
    /*
     * cost = 1/2 * || f(x) - y ||^2
     * cost of a single vector.  
     * @param x: input vector
     * @param input_size: length of x
     * @param hiddenlayers: model- function f.
     * @param numlayers: number of layers in model (3 in our case)
     * @param y: validation output vector.
     */
    float* output;
    forward_prop(x, input_size, hiddenlayers, numlayers);
    output = hiddenlayers[1].output;
    return 0.5*loss(output, y, hiddenlayers[1].n_out);
}

float loss(float* input, float* output, int n) {
    /* L2 regression loss function
     * L(f, (x,y)) = ||f(x) - y||^2
     * sum of squared element-wise differences
     * @param input: f(x) vector (predicted value)
     * @param output: y vector (actual value)
     * @param n: size of input and output vectors.
     */
    float sum = 0.0;
    for (int i = 0; i<n; i++) {
	sum += pow((input[i] - output[i]), 2);
    }
    return sum;
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

void testLoss() {
    // test the loss function
    float input[2] = {3, 3};
    float output[2] = {2, 2};
    float l = loss(input, output, 2); // should be (3-2) + (3-2) = 2
    assert(l==2.0);
    //printf("loss = %f \n", l);
    float input2[2] = {5.5, 6.5};
    float output2[2] = {3.0, 0.0};
    float l2 = loss(input2, output2, 2); // should be (5.5-3) + (6.5-0) = 9
    //printf("loss = %f \n", l2);
    assert(l2==9.0);
}

void testCost(Hidden_Layer* hiddenlayers) {
    //test the cost function with a simple model
    float input[36] = {0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01};
    float output[2] = {1.0 ,1.0};
    // cost1 should be .5*[(output[0] - model[0]) + (output[1] - model[1])] = .5*[(1-.5) + (1-.5)] = 0.5
    float cost1 = cost(input, 36, hiddenlayers, 3, output); 
    assert (cost1 == 0.5);
}
int main(int argc, char* argv[]){
    //testLoss();
    Hidden_Layer* hiddenlayers = new Hidden_Layer[2];
    int layer_sizes[3] = {36, 36, 2};
    init(3,layer_sizes, hiddenlayers);
    //testCost(hiddenlayers);
    //fread(weights, sizeof(float), 36*36, pFile);
    //fclose(pFile);
    float* output;
    float input[36] = {0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01};
    //cblas_sgemv(CblasRowMajor, CblasNoTrans, 2, 36, 1.0f, input2, 36, input, 1, 1.0f, output, 1);
    forward_prop(input, 36, hiddenlayers, 3);
    output = hiddenlayers[1].output;
    printf("%f  %f \n", output[0], output[1]);




};
