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

    tk::dnn::Conv2d conv1(&net, 32, 3, 3, 2, 2, 1, 1, "test_deer_hde/layers/conv1.bin", true);
    tk::dnn::Activation relu1(&net, CUDNN_ACTIVATION_RELU);

    // First depthwise block
    tk::dnn::Conv2d conv_dw_1(&net, 32, 3, 3, 1, 1, 1, 1, "test_deer_hde/layers/conv_dw_1.bin", true, false, 32);
    tk::dnn::Activation relu2(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d conv_pw_1(&net, 64, 1, 1, 1, 1, 0, 0, "test_deer_hde/layers/conv_pw_1.bin", true, false, 1);
    tk::dnn::Activation relu3(&net, CUDNN_ACTIVATION_RELU);

    // Second depthwise block
    tk::dnn::Conv2d conv_dw_2(&net, 64, 3, 3, 2, 2, 1, 1, "test_deer_hde/layers/conv_dw_2.bin", true, false, 64);
    tk::dnn::Activation relu4(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d conv_pw_2(&net, 128, 1, 1, 1, 1, 0, 0, "test_deer_hde/layers/conv_pw_2.bin", true, false, 1);
    tk::dnn::Activation relu5(&net, CUDNN_ACTIVATION_RELU);

    // Third depthwise block
    tk::dnn::Conv2d conv_dw_3(&net, 128, 3, 3, 1, 1, 1, 1, "test_deer_hde/layers/conv_dw_3.bin", true, false, 128);
    tk::dnn::Activation relu6(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d conv_pw_3(&net, 128, 1, 1, 1, 1, 0, 0, "test_deer_hde/layers/conv_pw_3.bin", true, false, 1);
    tk::dnn::Activation relu7(&net, CUDNN_ACTIVATION_RELU);

    // Fourth depthwise block
    tk::dnn::Conv2d conv_dw_4(&net, 128, 3, 3, 2, 2, 1, 1, "test_deer_hde/layers/conv_dw_4.bin", true, false, 128);
    tk::dnn::Activation relu8(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d conv_pw_4(&net, 256, 1, 1, 1, 1, 0, 0, "test_deer_hde/layers/conv_pw_4.bin", true, false, 1);
    tk::dnn::Activation relu9(&net, CUDNN_ACTIVATION_RELU);

    // Fifth depthwise block
    tk::dnn::Conv2d conv_dw_5(&net, 256, 3, 3, 1, 1, 1, 1, "test_deer_hde/layers/conv_dw_5.bin", true, false, 256);
    tk::dnn::Activation relu10(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d conv_pw_5(&net, 256, 1, 1, 1, 1, 0, 0, "test_deer_hde/layers/conv_pw_5.bin", true, false, 1);
    tk::dnn::Activation relu11(&net, CUDNN_ACTIVATION_RELU);

    // Sixth depthwise block
    tk::dnn::Conv2d conv_dw_6(&net, 256, 3, 3, 2, 2, 1, 1, "test_deer_hde/layers/conv_dw_6.bin", true, false, 256);
    tk::dnn::Activation relu12(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d conv_pw_6(&net, 512, 1, 1, 1, 1, 0, 0, "test_deer_hde/layers/conv_pw_6.bin", true, false, 1);
    tk::dnn::Activation relu13(&net, CUDNN_ACTIVATION_RELU);

    // Seventh depthwise block
    tk::dnn::Conv2d conv_dw_7(&net, 512, 3, 3, 1, 1, 1, 1, "test_deer_hde/layers/conv_dw_7.bin", true, false, 512);
    tk::dnn::Activation relu14(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d conv_pw_7(&net, 512, 1, 1, 1, 1, 0, 0, "test_deer_hde/layers/conv_pw_7.bin", true, false, 1);
    tk::dnn::Activation relu15(&net, CUDNN_ACTIVATION_RELU);

    // Eigth depthwise block
    tk::dnn::Conv2d conv_dw_8(&net, 512, 3, 3, 1, 1, 1, 1, "test_deer_hde/layers/conv_dw_8.bin", true, false, 512);
    tk::dnn::Activation relu16(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d conv_pw_8(&net, 512, 1, 1, 1, 1, 0, 0, "test_deer_hde/layers/conv_pw_8.bin", true, false, 1);
    tk::dnn::Activation relu17(&net, CUDNN_ACTIVATION_RELU);

    // Ninth depthwise block
    tk::dnn::Conv2d conv_dw_9(&net, 512, 3, 3, 1, 1, 1, 1, "test_deer_hde/layers/conv_dw_9.bin", true, false, 512);
    tk::dnn::Activation relu18(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d conv_pw_9(&net, 512, 1, 1, 1, 1, 0, 0, "test_deer_hde/layers/conv_pw_9.bin", true, false, 1);
    tk::dnn::Activation relu19(&net, CUDNN_ACTIVATION_RELU);

    // Tenth depthwise block
    tk::dnn::Conv2d conv_dw_10(&net, 512, 3, 3, 1, 1, 1, 1, "test_deer_hde/layers/conv_dw_10.bin", true, false, 512);
    tk::dnn::Activation relu20(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d conv_pw_10(&net, 512, 1, 1, 1, 1, 0, 0, "test_deer_hde/layers/conv_pw_10.bin", true, false, 1);
    tk::dnn::Activation relu21(&net, CUDNN_ACTIVATION_RELU);

    // Eleventh depthwise block
    tk::dnn::Conv2d conv_dw_11(&net, 512, 3, 3, 1, 1, 1, 1, "test_deer_hde/layers/conv_dw_11.bin", true, false, 512);
    tk::dnn::Activation relu22(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d conv_pw_11(&net, 512, 1, 1, 1, 1, 0, 0, "test_deer_hde/layers/conv_pw_11.bin", true, false, 1);
    tk::dnn::Activation relu23(&net, CUDNN_ACTIVATION_RELU);

    // Twelfth depthwise block
    tk::dnn::Conv2d conv_dw_12(&net, 512, 3, 3, 2, 2, 1, 1, "test_deer_hde/layers/conv_dw_12.bin", true, false, 512);
    tk::dnn::Activation relu24(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d conv_pw_12(&net, 1024, 1, 1, 1, 1, 0, 0, "test_deer_hde/layers/conv_pw_12.bin", true, false, 1);
    tk::dnn::Activation relu25(&net, CUDNN_ACTIVATION_RELU);

    // Twelfth depthwise block
    tk::dnn::Conv2d conv_dw_13(&net, 1024, 3, 3, 1, 1, 1, 1, "test_deer_hde/layers/conv_dw_13.bin", true, false, 1024);
    tk::dnn::Activation relu26(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d conv_pw_13(&net, 1024, 1, 1, 1, 1, 0, 0, "test_deer_hde/layers/conv_pw_13.bin", true, false, 1);
    tk::dnn::Activation relu27(&net, CUDNN_ACTIVATION_RELU);

    //Final layers 12
    tk::dnn::Conv2d conv2d(&net, 32, 3, 3, 3, 3, 0, 0, "test_deer_hde/layers/conv2d.bin");
    tk::dnn::Dense dense1(&net, 64, "test_deer_hde/layers/dense.bin");
    tk::dnn::Activation sigmoid1(&net, CUDNN_ACTIVATION_SIGMOID);
    tk::dnn::Dense dense2(&net, 2, "test_deer_hde/layers/out.bin");
    tk::dnn::Activation sigmoid2(&net, CUDNN_ACTIVATION_SIGMOID);
    tk::dnn::Layer *header[1] = {&sigmoid2};

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
