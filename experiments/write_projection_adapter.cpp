// Write the rank-one LoRA factors for Qwen2.5-0.5B's last attention output projection.
#include "ggml.h"
#include "gguf.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

int main(int argc, char **argv) try {
    constexpr int width = 896;
    constexpr int samples = 32;
    if (argc != 3 + samples) throw std::runtime_error("usage: write_projection_adapter OUTPUT STEP W0 ... W31");

    const int step = std::stoi(argv[2]);
    if (step == 0 || step < -8 || step > 8) throw std::runtime_error("invalid nonzero step");
    ggml_init_params params = {2 * ggml_tensor_overhead() + 4 * sizeof(float) * width, nullptr, false};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors(ggml_init(params), ggml_free);
    if (!tensors) throw std::runtime_error("tensor allocation failed");
    auto a = ggml_new_tensor_2d(tensors.get(), GGML_TYPE_F32, width, 1);
    auto b = ggml_new_tensor_2d(tensors.get(), GGML_TYPE_F32, 1, width);
    ggml_set_name(a, "blk.23.attn_output.weight.lora_a");
    ggml_set_name(b, "blk.23.attn_output.weight.lora_b");
    auto av = static_cast<float *>(a->data);
    auto bv = static_cast<float *>(b->data);
    std::fill_n(av, width, 0.0f);
    std::fill_n(bv, width, 0.0f);
    av[0] = 1.0f;
    for (int i = 0; i < samples; ++i) {
        const int coefficient = std::stoi(argv[3 + i]);
        if (coefficient < -16 || coefficient > 16) throw std::runtime_error("invalid frozen head coordinate");
        bv[i * (width / samples)] = static_cast<float>(step * coefficient) / 64.0f;
    }

    std::unique_ptr<gguf_context, decltype(&gguf_free)> file(gguf_init_empty(), gguf_free);
    if (!file) throw std::runtime_error("GGUF allocation failed");
    gguf_set_val_str(file.get(), "general.architecture", "qwen2");
    gguf_set_val_str(file.get(), "general.type", "adapter");
    gguf_set_val_str(file.get(), "adapter.type", "lora");
    gguf_set_val_f32(file.get(), "adapter.lora.alpha", 1.0f);
    gguf_add_tensor(file.get(), a);
    gguf_add_tensor(file.get(), b);
    if (!gguf_write_to_file(file.get(), argv[1], false)) throw std::runtime_error("GGUF write failed");
} catch (const std::exception &error) {
    std::cerr << error.what() << std::endl;
    return 1;
}
