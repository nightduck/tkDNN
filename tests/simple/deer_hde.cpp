#include<iostream>
#include "tkdnn.h"

const char *output_bin = "test_deer_hde/debug/output.bin";
const char *input_bin = "test_deer_hde/debug/input.bin";

const char *conv0_bin = "test_deer_hde/layers/conv2d.bin";
const char *inverted_residual1[] = {
    "test_deer_hde/layers/depthwise_conv2d.bin",
    "test_deer_hde/layers/depthwise_conv2d_1.bin"};
const char *inverted_residual2[] = {
    "test_deer_hde/layers/depthwise_conv2d_2.bin",
    "test_deer_hde/layers/depthwise_conv2d_3.bin"};
const char *inverted_residual3[] = {
    "test_deer_hde/layers/depthwise_conv2d_4.bin",
    "test_deer_hde/layers/depthwise_conv2d_5.bin"};
const char *inverted_residual4[] = {
    "test_deer_hde/layers/depthwise_conv2d_6.bin",
    "test_deer_hde/layers/depthwise_conv2d_7.bin"};
const char *inverted_residual5[] = {
    "test_deer_hde/layers/depthwise_conv2d_8.bin",
    "test_deer_hde/layers/depthwise_conv2d_9.bin"};
const char *conv12_name = "test_deer_hde/layers/conv2d_1.bin";
const char *dense_name = "test_deer_hde/layers/out.bin";

int main() {

        // Network layout
    tk::dnn::dataDim_t dim(1, 3, 192, 192, 1);
    tk::dnn::Network net(dim);

    tk::dnn::Conv2d conv1(&net, 32, 3, 3, 2, 2, 1, 1, conv0_bin);
    tk::dnn::Activation relu1(&net, CUDNN_ACTIVATION_RELU);

    // First group of depthwise
    tk::dnn::Conv2d conv2(&net, 32, 3, 3, 1, 1, 1, 1, inverted_residual1[0], false, false, 32);
    tk::dnn::Activation relu2(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d conv3(&net, 32, 3, 3, 2, 2, 1, 1, inverted_residual1[1], false, false, 32);
    tk::dnn::Activation relu3(&net, CUDNN_ACTIVATION_RELU);

    // Second group of depthwise
    tk::dnn::Conv2d conv4(&net, 32, 3, 3, 1, 1, 1, 1, inverted_residual2[0], false, false, 32);
    tk::dnn::Activation relu4(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d conv5(&net, 32, 3, 3, 2, 2, 1, 1, inverted_residual2[1], false, false, 32);
    tk::dnn::Activation relu5(&net, CUDNN_ACTIVATION_RELU);

    // Third group of depthwise
    tk::dnn::Conv2d conv6(&net, 32, 3, 3, 1, 1, 1, 1, inverted_residual3[0], false, false, 32);
    tk::dnn::Activation relu6(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d conv7(&net, 32, 3, 3, 2, 2, 1, 1, inverted_residual3[1], false, false, 32);
    tk::dnn::Activation relu7(&net, CUDNN_ACTIVATION_RELU);

    // Fourth group of depthwise
    tk::dnn::Conv2d conv8(&net, 32, 3, 3, 1, 1, 1, 1, inverted_residual4[0], false, false, 32);
    tk::dnn::Activation relu8(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d conv9(&net, 32, 3, 3, 2, 2, 1, 1, inverted_residual4[1], false, false, 32);
    tk::dnn::Activation relu9(&net, CUDNN_ACTIVATION_RELU);

    // Fifth group of depthwise
    tk::dnn::Conv2d conv10(&net, 32, 3, 3, 1, 1, 1, 1, inverted_residual5[0], false, false, 32);
    tk::dnn::Activation relu10(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d conv11(&net, 32, 3, 3, 2, 2, 1, 1, inverted_residual5[1], false, false, 32);
    tk::dnn::Activation relu11(&net, CUDNN_ACTIVATION_RELU);

    //Final layers 12
    tk::dnn::Conv2d conv12(&net, 32, 1, 1, 1, 1, 0, 0, conv12_name);
    tk::dnn::Activation relu12(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Dense dense(&net, 2, dense_name);
    tk::dnn::Activation sigmoid1(&net, CUDNN_ACTIVATION_SIGMOID);
    tk::dnn::Layer *header[1] = {&sigmoid1};

    net.print();

    // Load input
    dnnType *data;
    dnnType *input_h;
    readBinaryFile(input_bin, dim.tot(), &input_h, &data);

    // // Print input
    // std::cout<<"\n======= INPUT =======\n";
    // printDeviceVector(dim.tot(), data);
    // std::cout<<"\n";

    //convert network to tensorRT
    tk::dnn::NetworkRT netRT(&net, net.getNetworkRTName("deer_hde"));

    dnnType *out_data, *out_data2; // cudnn output, tensorRT output

    tk::dnn::dataDim_t dim1 = dim; //input dim
    printCenteredTitle(" CUDNN inference ", '=', 30); {
        dim1.print();
        TKDNN_TSTART
        out_data = net.infer(dim1, data);
        TKDNN_TSTOP
        dim1.print();
    }

    tk::dnn::dataDim_t dim2 = dim;
    printCenteredTitle(" TENSORRT inference ", '=', 30); {
        dim2.print();
        TKDNN_TSTART
        out_data2 = netRT.infer(dim2, data);
        TKDNN_TSTOP
        dim2.print();
    }

    std::cout<<"\n======= CUDNN =======\n";
    printDeviceVector(dim1.tot(), out_data);
    std::cout<<"\n======= TENSORRT =======\n";
    printDeviceVector(dim2.tot(), out_data2);

    printCenteredTitle(" CHECK RESULTS ", '=', 30);
    dnnType *out, *out_h;
    int out_dim = net.getOutputDim().tot();

    readBinaryFile(output_bin, out_dim, &out_h, &out);
    std::cout<<"CUDNN vs correct"; 
    int ret_cudnn = checkResult(out_dim, out_data, out) == 0 ? 0: ERROR_CUDNN;
    std::cout<<"TRT   vs correct"; 
    int ret_tensorrt = checkResult(out_dim, out_data2, out) == 0 ? 0 : ERROR_TENSORRT;

    std::cout<<"CUDNN vs TRT    "; 
    int ret_cudnn_tensorrt = checkResult(out_dim, out_data, out_data2) == 0 ? 0 : ERROR_CUDNNvsTENSORRT;

    return ret_cudnn_tensorrt;
}
