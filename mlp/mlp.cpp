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
void softmax_prime(float* input, float* output, int n);
float cost(float* x, int input_size, Hidden_Layer* hiddenlayers, int numlayers, float* y);
struct delta;
void backprop(int input, int input_size, Hidden_Layer* hiddenlayers, int numlayers, float* actual, delta& deltas);
float tanh_prime(float input);
int store_weights(float* weights, int n);
int read_weights(float* weights, int n);
class Hidden_Layer {
    /* class definition that represents a hidden layer. On construction computes the hidden layer
     *  represented by the layer weights, bias vector, induced local field vector (Wx+b) and output.   
     *  n_in should be the #elements in input vector and #cols of layer_weights.
     *  n_out is the number of hidden layers and #rows of layer_weights.
     */
 public: 
    float *layer_weights, *bias, *output, *v;
    int n_in, n_out;
    Hidden_Layer(){};
    Hidden_Layer(float *, float *, int, int);
    void compute_output(float* input);
    void update_weights(float* deltas, float learnrate);
    void update_bias(float* deltas, float learnrate);
};
Hidden_Layer::Hidden_Layer(float* weights, float* b, int in, int out) {
    layer_weights = weights;
    bias = b;
    n_in = in;
    n_out = out;
    output = (float *) malloc(n_out*sizeof(float));
    if (!output) {
	exit(1);
    }
    v = (float *) malloc(n_out*sizeof(float)); 
}
void Hidden_Layer::compute_output(float* input) {
    printf("nout:%d, nin:%d \n", n_out, n_in);
    printf("OUTPUT BEFORE SGEMV = ["); for(int i=0; i<n_out; i++){ printf("%f, ", output[i]); } printf( " ]\n");

    printf("input: \n[ "); for(int i=0; i<36; i++){ printf("%3f, ", input[i]); } printf("]\n");
    float* output2 = (float*) malloc(36*sizeof(float)); if(!output2){exit(1);}
    output[3] = 333.33;
    cblas_sgemv(CblasRowMajor, CblasNoTrans, n_out, n_in, 1.0f, layer_weights, n_in, input, 1, 1.0f, output2, 1); //computes Wx
    memcpy(output, output2, 36*sizeof(float));
    free(output2);
    printf("OUTPUT AFTER  SGEMV = [");
    for(int i=0; i<n_out; i++){
	printf("%f, ", output[i]);
    }
    printf( " ]\n");

    cblas_saxpy(n_out, 1.0, bias, 1, output, 1);


    for (int i = 0; i < n_out; i++) {
    	v[i] = output[i];
        output[i] = tanh(output[i]);
    }


}   
void Hidden_Layer::update_weights(float* deltas, float learnrate) {
    for (int i = 0; i < n_out*n_in; i++) {
        layer_weights[i] -= learnrate*deltas[i];
    }
}

void Hidden_Layer::update_bias(float* deltas, float learnrate) {
    for (int i = 0; i < n_out; i++) {
        bias[i] -= learnrate*deltas[i];
    }
}

int store_weights(float* weights, int n) {
    FILE *weightsfile;
    weightsfile = fopen("weights", "wb");
    if (weightsfile == NULL) return 0;
    fwrite(weights, sizeof(float), n, weightsfile);
    return 1; 
}

int read_weights(float* weights, int n) {
    FILE *weightsfile;
    weightsfile = fopen("weights", "rb");
    if (weightsfile == NULL) return 0;
    fread(weights, sizeof(float), n, weightsfile);
    printf("WEIGHTSFILE EXIST\n\n\n\n\n");
    return 1;
}

void forward_prop(float *input, int input_size, Hidden_Layer* hiddenlayers, int numlayers) {
    /* @param input: array of length 36 representing 6 6x1 vectors
     * @param input_size: size of input array.
     * @param weights: weight vector
     * @param dim: array of layer sizes.  dim[0] is size of input, dim[1] is size of 1st hidden layer.   
     * 	       	   dim[dim_size-1] is 1 for our binary classifier.
     * @param sim_size: size of dim array
     */
    float* input_tanh = (float*) malloc(input_size*sizeof(float));
    if ( !input_tanh ) {
	exit(1);
    }
    printf("before memcpy: \n[");
    for(int i=0; i<input_size; i++){
//	printf("%5f ,", input[i]);
    }
    printf(" ]\n");
    memcpy(input_tanh, input, input_size*sizeof(float));

    printf("after memcpy: \n[");
    for(int i=0; i<input_size; i++){
//	printf("%5f ,", input_tanh[i]);
    }
    printf(" ]\n");
    hiddenlayers[0].compute_output(input_tanh);
    for (int i = 1; i < numlayers-1; i++) {
	   hiddenlayers[i].compute_output(hiddenlayers[i-1].output);
    }
    //free(input_tanh);
};

struct delta{
    float* input_error;
    float* bias1;
    float* bias2;
    float* weights1;
    float* weights2;
};

void backprop(float* input, int input_size, Hidden_Layer* hiddenlayers, int numlayers, float* actual, delta& deltas) {
    //float cost1 = cost(input, input_size, hiddenlayers, numlayers, actual); // calls forward_prop
    forward_prop(input, input_size, hiddenlayers, numlayers);
    int layer_sizes[numlayers]; // [300,300, 2]
    layer_sizes[0] = hiddenlayers[0].n_in;
    for(int i=0; i<numlayers-1; i++){
    	layer_sizes[i+1] = hiddenlayers[i].n_out;
    }
    if ( !(deltas.bias1 && deltas.bias2 && deltas.weights1 && deltas.weights2 && deltas.input_error) ) { exit(1); }

    //final error is (a_L - y) (*) tanh'(Wx+b), where (*) is the Hadamard (element wise product)
    for(int i=0; i<hiddenlayers[numlayers-2].n_out; i++) {
	   deltas.bias2[i] = hiddenlayers[numlayers-2].output[i];
	   printf( "deltas.bias2[%d] = %f\n", i, deltas.bias2[i]);
    }
    int output_size = hiddenlayers[numlayers-2].n_out; //300
    // float activation_partial[output_size];
    //softmax_prime(hiddenlayers[numlayers-2].v, activation_partial, output_size); // softmax'(Wx+b)
    for(int i=0; i<hiddenlayers[numlayers-2].n_out; i++) {
	   deltas.bias2[i] -= actual[i]; // a_L - y
	   deltas.bias2[i] *= tanh_prime(hiddenlayers[numlayers-2].v[i]); // errors[1] is output layer errors
    }
    // next compute error for all layers (layer 1-input and 2-hidden). error_l = ((w_l+1)^T * error_l+1) (*) tanh'(Wx+b)
    // do layer 2 (bias1).  first find (w_i+1)^T * errors[i]
    cblas_sgemv(CblasRowMajor, CblasTrans, 
	    layer_sizes[2], // M = 2
	    layer_sizes[1], // N = 300
	    1.0f, //
	    hiddenlayers[1].layer_weights, //weights between this layer and the next
	    layer_sizes[1], // lda = 300
	    deltas.bias2, //error at the next layer
	    1,
	    1.0f,
	    deltas.bias1, // output
	    1);
    for(int j = 0; j < hiddenlayers[1].n_in; j++) { // 0 to 300
	deltas.bias1[j] *= tanh_prime(hiddenlayers[0].v[j]);
    }
    
    // next do layer 1 (input_error).  first find (w_i+1)^T * errors[i]
    cblas_sgemv(CblasRowMajor, CblasTrans, 
	    layer_sizes[1], // M = 300
	    layer_sizes[0], // N = 300
	    1.0f, //
	    hiddenlayers[0].layer_weights, //weights between this layer and the next
	    layer_sizes[0], // lda = 300
	    deltas.bias1, //error at the next layer
	    1,
	    1.0f,
	    deltas.input_error, // output
	    1);
    // for the input layer, the activation is just the input vector
    for(int j = 0; j < hiddenlayers[0].n_in; j++) { // 0 to 300
	deltas.input_error[j] *= tanh_prime(input[j]);
    }

    // partial derivitive of C wrt to bias is just error.
    // compute partial derivative of C wrt weights. d = a_(l-1) * error_l
    // compute deltas.weights	

    // dot product of activation of hidden layer (layer 2) with the error of output layer (layer 3)
    cblas_sgemm(CblasRowMajor,
	    CblasNoTrans,
	    CblasNoTrans,
	    hiddenlayers[1].n_in,  //M 300
	    hiddenlayers[1].n_out, //N (2)
	    1, //K=1
	    1.0f, //alpha
	    hiddenlayers[0].output, //matrix A, the activation of layer 2 (size 300 by 1) 
	    1, //lda = 1 (don't ask me why)
	    deltas.bias2, //(size 1 by 2)
	    2, //ldb=2
	    1.0f, //beta
	    deltas.weights2, //output
	    2//ldc=2
	    );
    float* tanh_input = (float*) malloc(input_size*sizeof(float));
    if ( !tanh_input ) {exit(1);};
    for ( int i=0; i<input_size; i++) {
	tanh_input[i] = tanh(input[i]);
    }
    // now compute the partial weights between layer 1 and 2->   activation_1 * error_2
    cblas_sgemm(CblasRowMajor,
	    CblasNoTrans,
	    CblasNoTrans,
	    hiddenlayers[0].n_in,  //M (300)
	    hiddenlayers[0].n_out, //N (300)
	    1, //K=1
	    1.0f, //alpha
	    tanh_input, //matrix A, the activation of layer 1 (size 300 by 1).  i.e. tanh(input)
	    1, //lda = 1 (don't ask me why)
	    deltas.bias1, //(size 300 by 1)
	    hiddenlayers[0].n_out, //
	    1.0f, //beta
	    deltas.weights1, //output (300 by 300)
	    hiddenlayers[0].n_out//
	    );

}

float tanh_prime(float input) {
    return (4*pow(cosh(input), 2)) / pow((cosh(2*input)+1), 2);
}
void softmax_prime(float* input, float* output, int n) {
    /* Computes derivative of softmax. Softmax prime takes in a vector and outputs a square jacobian matrix
     * @param input: input vector of length n
     * @param output: output matrix of size n*n
     * @param n: size of input vector
     */
    float * interoutput = (float *) malloc(n*sizeof(float));
    softmax(input, interoutput, n);
    for (int i = 0; i < n; i++) {
	for (int k = 0; k < n; k++) {
	    float delta = 0; 
	    if (i == k) delta = 1;
	    output[k+n*i] = interoutput[i] * (delta - interoutput[k]);
	}
    }
    //free(interoutput);
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
    if (!read_weights(weights, weights_size)) {
        srand (time(NULL));
        for (int i=0; i<weights_size; i++){
	   //weights[i] = float(float(rand()%10)/float(10));
            weights[i] = 0.5;
        }
        for (int i = 0; i < bias_size; i++) {
            //bias[i] = float(float(rand()%10)/float(10));
            bias[i] = 0.5;
        }
    }
    int weights_offset = 0;
    int bias_offset = 0;
    for (int j = 0; j < numlayers-1; j++) {
        Hidden_Layer temp(weights+weights_offset, bias+bias_offset, layer_sizes[j], layer_sizes[j+1]);
        hiddenlayers[j] = temp;
        weights_offset += layer_sizes[j] * layer_sizes[j+1];
        bias_offset += layer_sizes[j];
    }
};

void testsoftmaxprime() {
    float input[2] = {1.0, 0.8};
    float output1[4];
    softmax_prime(input, output1, 2);
    for (int i = 0; i < 4; i++) {
        printf("softmax prime : %f   ", output1[i]);
    }
}
void testLoss() {
    // test the loss function
    float input[2] = {3, 3};
    float output[2] = {2, 2};
    float l = loss(input, output, 2); // should be (3-2) + (3-2) = 2
    assert(l==2.0);
    ////printf("loss = %f \n", l);
    float input2[2] = {5.5, 6.5};
    float output2[2] = {3.0, 0.0};
    float l2 = loss(input2, output2, 2); // should be (5.5-3) + (6.5-0) = 9
    ////printf("loss = %f \n", l2);
    assert(l2==9.0);
}

void testCost(Hidden_Layer* hiddenlayers) {
    //test the cost function with a simple model
    float input[36] = {100.0, 333.33, 333.55, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01};
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
    float* output;
    float input[36] = {0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01};
    float* input2 = (float*) malloc(36*sizeof(float));
    for(int i=0; i<36; i++){
	   input2[i] = 1 /  (i+1);
    }
    //cblas_sgemv(CblasRowMajor, CblasNoTrans, 2, 36, 1.0f, input2, 36, input, 1, 1.0f, output, 1);
    output = hiddenlayers[1].output;
    //forward_prop(input, 36, hiddenlayers, 3);

    float realoutput[2] = { 1, 0 };

    delta deltas;
    deltas.bias1 = (float*) malloc(layer_sizes[0]*sizeof(float));
    deltas.bias2 = (float*) malloc(layer_sizes[1]*sizeof(float));
    deltas.weights1 = (float*) malloc(layer_sizes[0]*layer_sizes[1]*sizeof(float));
    deltas.weights2 = (float*) malloc(layer_sizes[1]*layer_sizes[2]*sizeof(float));
    deltas.input_error = (float*) malloc(layer_sizes[0]*sizeof(float));
    
    backprop(input2, 36, hiddenlayers, 3, realoutput, deltas);
//void forward_prop(float *input, int input_size, Hidden_Layer* hiddenlayers, int numlayers) {
//    forward_prop(input2, 36, hiddenlayers, 3);

    printf("deltas: \n");
    for(int i=0; i<36; i++) {
	printf(" input offset %3d = %5f  |  bias1 offset = %5f  |  bias2 offset = %5f\n", i, deltas.input_error[i], deltas.bias1[i],deltas.bias2[i]);
    }






/*

    float input[36] = {0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01};
    float* output = (float *) malloc(36*sizeof(float));
    if (!output){exit(1);}
    int weights_size = 36*36+36*2;
    float* weights = (float*) malloc(weights_size*sizeof(float));
    if(!weights){exit(1);}
    if (!read_weights(weights, weights_size)) {
        srand (time(NULL));
        for (int i=0; i<weights_size; i++){
            weights[i] = 0.5;
        }
    }
    float* input3 = (float *) malloc(36*sizeof(float));
    if(!input3){exit(1);}
    for(int i=0; i<36; i++){
	input3[i] = tanh(input[i]);
    }
    printf("INPUT AFTER TANH = ["); for(int i=0; i<36; i++){ printf("%f, ", input3[i]); } printf( " ]\n");

    cblas_sgemv(CblasRowMajor, CblasNoTrans, 36, 36, 1.0f, weights, 36, input3, 1, 1.0f, output, 1); //computes Wx

    printf("OUTPUT AFTER  SGEMV = ["); for(int i=0; i<36; i++){ printf("%f, ", output[i]); } printf( " ]\n");
*/

};
