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
void backprop(int input, int input_size, Hidden_Layer* hiddenlayers, int numlayers, float* actual);
float tanh_prime(float input);

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
    void compute_output(float* input, int last_layer);

};
Hidden_Layer::Hidden_Layer(float* weights, float* b, int in, int out) {
    layer_weights = weights;
    bias = b;
    n_in = in;
    n_out = out;
    output = (float *) malloc(n_out*sizeof(float));
    v = (float *) malloc(n_out*sizeof(float)); 
}
void Hidden_Layer::compute_output(float* input, int last_layer) {
    //printf("nout:%d, nin:%d \n", n_out, n_in);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, n_out, n_in, 1.0f, layer_weights, n_in, input, 1, 1.0f, output, 1); //computes Wx

    cblas_saxpy(n_out, 1.0, bias, 1, output, 1);
    if (!last_layer) {
    	for (int i = 0; i < n_out; i++) {
    	    v[i] = output[i];
    	    output[i] = tanh(output[i]);
        }
    } else {
    	for (int i = 0; i < n_out; i++) {
    	    v[i] = output[i];
        }
	    softmax(output, output, n_out);
    }
    //printf("OUTPUT = [%f, %f]\n", output[0], output[1]);
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

void backprop(float* input, int input_size, Hidden_Layer* hiddenlayers, int numlayers, float* actual) {
    float cost1 = cost(input, input_size, hiddenlayers, numlayers, actual);
    int largest_layer_size = hiddenlayers[0].n_in; //36
    int layer_sizes[numlayers]; // [36, 2]
    layer_sizes[0] = hiddenlayers[0].n_in;
    for(int i=0; i<numlayers-1; i++){
    	largest_layer_size = max(largest_layer_size, hiddenlayers[numlayers-i-2].n_out);
    	layer_sizes[i+1] = hiddenlayers[i].n_out;
    }
    float *errors = (float *) malloc((numlayers-1)*largest_layer_size*sizeof(float));
    //final error = (a_L - y) (*) softmax'(Wx+b), where (*) is the Hadamard (element wise product)
    for(int i=0; i<hiddenlayers[numlayers-2].n_out; i++) {
	   errors[(numlayers-2)*largest_layer_size+i] = hiddenlayers[numlayers-2].output[i];
    }
    int output_size = hiddenlayers[numlayers-2].n_out;
    float activation_partial[output_size];
    softmax_prime(hiddenlayers[numlayers-2].v, activation_partial, output_size); // softmax'(Wx+b)
    printf("softmaxprime      input 	output:       layer 1\n");
    printf("	0 		%f 	   %f    \n", hiddenlayers[numlayers-2].v[0], activation_partial[0]);  
    printf("	1 		%f 	   %f    \n", hiddenlayers[numlayers-2].v[1], activation_partial[1]);  
    for(int i=0; i<hiddenlayers[numlayers-2].n_out; i++) {
	   errors[(numlayers-2)*largest_layer_size+i] -= actual[i]; //a_L - y [-.5, .5]
	   ////printf("a_L - y for layer %d index %d = %f \n",numlayers-2, i, errors[numlayers-2][i]); 
	   errors[(numlayers-2)*largest_layer_size+i] *= activation_partial[i]; // errors[1] is output layer errors
    }
    // next compute error for all layers
    // error_l = ((w_l+1)^T * error_l+1) (*) softmax'(Wx+b)
    for(int i = numlayers-2; i>0; i--) { //only does i=1
	// first find (w_i+1)^T * errors[i]
	for(int j = 0; j < 2; j++) { // 0 to 36
	    //printf("INPUT = %f\n", errors[i*largest_layer_size+j]);
	}
	for(int j = 0; j < 72; j++) { // 0 to 36
	    //printf("LAYER_WEIGHTS_INPUT = %f\n", hiddenlayers[i].layer_weights[j]);
	}
	cblas_sgemv(CblasRowMajor, CblasTrans, 
		2, // M
		36, // N
		1.0f, //
		hiddenlayers[i].layer_weights,
		36,
		errors+largest_layer_size,
		1,
		1.0f,
		//errors[i-1], // output
		errors, // output
		1);
	for(int j = 0; j < 36; j++) { // 0 to 36
	    //printf("AFTER errors[%d][%d] = %f\n", 0, j, errors[j]);
	}
	for(int j = 0; j < hiddenlayers[i].n_in; j++) { // 0 to 36
	    ////printf("AFTER errors[%d][%d] = %f\n", i-1, j, output2[j]);
	    //errors[i-1][j] *= tanh_prime(hiddenlayers[i-1].v[j]);
	    errors[(i-1)*largest_layer_size + j] = errors[j] * tanh_prime(hiddenlayers[i-1].v[j]);
	}
    }
    printf("softmaxprime      input 	output:        layer 0\n");
    printf("	0 		%f 	       \n", hiddenlayers[numlayers-3].v[0]);  
    printf("	1 		%f 	       \n", hiddenlayers[numlayers-3].v[1]);  

    // partial derivitive of C wrt to bias is just error.
    // compute partial derivative of C wrt weights. d = a_(l-1) * error_l
    float* partialW1 = (float*) malloc(36*36*sizeof(float));
    float* partialW2 = (float*) malloc(36*2*sizeof(float));
    printf( "error at output = [ %f, %f ] \n", *(errors+36), *(errors+37));
    // dot product of activation of hidden layer (layer 2) with the error of output layer (layer 3)
    cblas_sgemm(CblasRowMajor,
	    CblasNoTrans,
	    CblasNoTrans,
	    hiddenlayers[1].n_in,  //M (36)
	    hiddenlayers[1].n_out, //N (2)
	    1, //K=1
	    1.0f, //alpha
	    hiddenlayers[0].output, //matrix A, the activation of layer 2
	    1, //lda = 1 (don't ask me why)
	    errors+36,
	    2, //ldb=2
	    1.0f, //beta
	    partialW2, //output
	    2//ldc=2
	    );
    printf( "output of matrix matrix: \n");
    for(int i=0; i<36; i++){
       printf(" [ %5f, %5f ]\n", partialW2[i], partialW2[i+36]);
    }       

    // now compute the partial weights between layer 1 and 2->   activation_1 * error_2
    printf( "error at hidden layer: \n");
    for(int i=0; i<36; i++){
	printf(" %f, ", *(errors+i));
    }
    // dot product of activation of hidden layer (layer 2) with the error of output layer (layer 3)
    cblas_sgemm(CblasRowMajor,
	    CblasNoTrans,
	    CblasNoTrans,
	    hiddenlayers[1].n_in,  //M (36)
	    hiddenlayers[1].n_out, //N (2)
	    1, //K=1
	    1.0f, //alpha
	    hiddenlayers[0].output, //matrix A, the activation of layer 2
	    1, //lda = 1 (don't ask me why)
	    errors+36,
	    2, //ldb=2
	    1.0f, //beta
	    partialW2, //output
	    2//ldc=2
	    );
    printf( "output of matrix matrix: \n");
    for(int i=0; i<36; i++){
       printf(" [ %5f, %5f ]\n", partialW2[i], partialW2[i+36]);
    }       


}

float tanh_prime(float input) {
    return (4*pow(cosh(input), 2)) / pow((cosh(2*input)+1), 2);
}
void softmax_prime(float* input, float* output, int n) {


/*    float denom = 0.0;
    float denom_prime = 0.0;
    for (int i=0; i<n; i++) {
        denom += exp(input[i]);
        denom_prime += input[i]*exp(input[i]);
    }
    float denomsq = pow(denom, 2);
    for (int i=0; i<n; i++) {
        output[i] = (input[i] * exp(input[i]) * denom - denom_prime*exp(input[i])) / denomsq;
    }
    */
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
        //printf("weights: %f ", *(weights+weights_offset));
        //printf("temp layer: %f \n", temp.layer_weights[0]);
        hiddenlayers[j] = temp;
        weights_offset += layer_sizes[j] * layer_sizes[j+1];
        bias_offset += layer_sizes[j];
    }
    //printf("\n hiddenlayer[1] weights: \n");
    for(int i=0; i<72; i++) {
	//printf("hiddenlayer[1].weights[%d] = %f\n", i, hiddenlayers[1].layer_weights[i]);
    }
    //printf("bias\n");
    //printf("bias[0] = %f\nbias[1] = %f", hiddenlayers[1].bias[0], hiddenlayers[1].bias[1]);
    //    FILE *pFile = fopen("weights", "wb");
    //    if (!pFile) {
	// std::cout<<"error opening file";
	// exit(0);
    //    };
    //    fwrite(weights, sizeof(float), 36*36, pFile);
    //    fclose(pFile); 
};

void testsoftmaxprime() {
    float input[36] = {0.23, 0.01, 0.13, 0.01, 0.45, 0.65, 0.14, 0.67, 0.01, 0.12, 0.01, 0.14, 0.98, 0.56, 0.72, 0.24, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01};
    float output1[36];
    softmax_prime(input, output1, 36);
    for (int i = 0; i < 36; i++) {
        ////printf("%f   ", output1[i]);
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
    float input[36] = {333.0, 333.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01};
    //testsoftmaxprime();
    //cblas_sgemv(CblasRowMajor, CblasNoTrans, 2, 36, 1.0f, input2, 36, input, 1, 1.0f, output, 1);
//    forward_prop(input, 36, hiddenlayers, 3);
    output = hiddenlayers[1].output;
    //printf("%f  %f \n", output[0], output[1]);

    //void backprop(float* input, int input_size, Hidden_Layer* hiddenlayers, int numlayers, float* actual) {
    float realoutput[2] = { 1, 0 };
    backprop(input, 36, hiddenlayers, 3, realoutput);



};
