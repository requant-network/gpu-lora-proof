// Capture the last-token final hidden row of each public text in isolation.
#include "ggml-backend.h"
#include "llama.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

int main(int argc, char **argv) try {
    require(argc >= 4 && argc <= 6, "usage: text_head_features MODEL LIBDIR CORPUS [--pinned-rtx|--cpu [ADAPTER]]");
    const std::string profile = argc >= 5 ? argv[4] : "";
    const bool cpu = profile == "--cpu";
    require(profile.empty() || cpu || profile == "--pinned-rtx", "unknown capture profile");
    std::ifstream corpus(argv[3]);
    require(corpus.good(), "corpus not readable");
    if (profile.empty()) {
        ggml_backend_load_all_from_path(argv[2]);
    } else {
        const std::string directory = argv[2];
        require(ggml_backend_load((directory + "/libggml-cpu-haswell.so").c_str()), "CPU backend failed to load");
        if (!cpu) {
            require(ggml_backend_load((directory + "/cuda_v13/libggml-cuda.so").c_str()), "CUDA backend failed to load");
        }
    }
    llama_backend_init();
    auto device = ggml_backend_dev_by_name("CUDA0");
    if (!cpu) {
        require(device && std::string(ggml_backend_dev_description(device)).find("RTX 5060 Ti") != std::string::npos,
                "expected the selected RTX 5060 Ti");
    }
    ggml_backend_dev_t devices[] = {device, nullptr};
    auto mp = llama_model_default_params();
    mp.devices = cpu ? nullptr : devices;
    mp.n_gpu_layers = cpu ? 0 : 99;
    mp.split_mode = LLAMA_SPLIT_MODE_NONE;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(argv[1], mp), llama_model_free);
    require(bool(model), "model load failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = 1024;
    cp.n_batch = cp.n_ubatch = 256;
    cp.n_seq_max = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cp.embeddings = true;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
    require(bool(ctx), "context creation failed");
    if (argc == 6) {
        require(cpu, "the projection adapter fixture runs on the CPU profile");
        auto adapter = llama_adapter_lora_init(model.get(), argv[5]);
        require(adapter, "adapter load failed");
        float scale = 1.0f;
        require(llama_set_adapters_lora(ctx.get(), &adapter, 1, &scale) == 0, "adapter activation failed");
    }
    auto vocab = llama_model_get_vocab(model.get());
    const int width = llama_model_n_embd_out(model.get());
    require(width == 896, "unexpected frozen embedding width");
    std::cout << "width\t" << width << '\n' << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (std::string line; std::getline(corpus, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        auto split = line.find('\t');
        require(split != std::string::npos, "missing split delimiter");
        auto id = line.find('\t', split + 1);
        require(id != std::string::npos, "missing ID delimiter");
        auto label = line.find('\t', id + 1);
        require(label != std::string::npos && label + 1 < line.size(), "missing text");
        const auto prompt = std::string("<|im_start|>system\nClassify the following news brief by topic: sports or weather.<|im_end|>\n") +
            "<|im_start|>user\n" + line.substr(label + 1) + "<|im_end|>\n<|im_start|>assistant\n";
        std::vector<llama_token> tokens(256);
        const int n = llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), tokens.size(), true, true);
        require(n > 0 && n <= int(tokens.size()), "unexpected token count");
        llama_memory_clear(llama_get_memory(ctx.get()), true);
        require(llama_decode(ctx.get(), llama_batch_get_one(tokens.data(), n)) == 0, "feature decode failed");
        llama_synchronize(ctx.get());
        const float *row = llama_get_embeddings_ith(ctx.get(), -1);
        require(row, "missing final hidden row");
        std::cout << line.substr(split + 1, id - split - 1) << '\t' << n;
        for (int i = 0; i < 32; ++i) {
            const float value = row[i * width / 32];
            require(std::isfinite(value), "nonfinite hidden component");
            std::cout << '\t' << value;
        }
        std::cout << '\n';
    }
    require(corpus.eof() && std::cout.good(), "feature capture incomplete");
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
}
