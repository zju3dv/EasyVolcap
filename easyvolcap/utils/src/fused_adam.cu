#include <math.h>
#include <torch/extension.h>

__global__ void adam_kernel(
    float* param,
    const float* grad,
    float* exp_avg,
    float* exp_avg_sq,
    const float step,
    const float beta1,
    const float beta2,
    const float lr,
    const float eps,
    const int P) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < P && grad[i] != 0.0) {
    // if (i < P) {
        // Decay the first and second moment running average coefficient
        exp_avg[i] = exp_avg[i] * beta1 + (1.0 - beta1) * grad[i];
        exp_avg_sq[i] = exp_avg_sq[i] * beta2 + (1.0 - beta2) * grad[i] * grad[i];

        float bias_correction1 = 1.0 - powf(beta1, step);
        float bias_correction2 = 1.0 - powf(beta2, step);

        float step_size = lr / bias_correction1;
        float bias_correction2_sqrt = sqrtf(bias_correction2);
        float denom = sqrtf(exp_avg_sq[i]) / bias_correction2_sqrt + eps;

        // Update parameters
        param[i] -= (exp_avg[i] / denom) * step_size;
    }
}

extern "C" void fused_adam(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    const float step_t,  // already updated
    const float beta1,
    const float beta2,
    const float lr,
    const float eps) {
    int blocks, threads;
    int P = param.numel();
    if (P > 0) {
        if (P > 256) {
            blocks = (P + 256 - 1) / 256;
            threads = 256;
        } else {
            blocks = 1;
            threads = P;
        }
        // Call CUDA kernel
        adam_kernel<<<blocks, threads>>>(
            param.data_ptr<float>(),
            grad.data_ptr<float>(),
            exp_avg.data_ptr<float>(),
            exp_avg_sq.data_ptr<float>(),
            step_t,  // Pass step as a float
            beta1,
            beta2,
            lr,
            eps,
            P);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_adam", &fused_adam, "Adam update (CUDA)");
}