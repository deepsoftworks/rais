#include <rais/scheduler.hpp>

#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

struct Request {
    int id;
    std::string prompt;
    rais::Lane lane;
};

static const char* lane_name(rais::Lane lane) {
    switch (lane) {
        case rais::Lane::Interactive: return "Interactive";
        case rais::Lane::Background: return "Background";
        case rais::Lane::Bulk: return "Bulk";
        case rais::Lane::GPU: return "GPU";
        case rais::Lane::IO: return "IO";
    }
    return "Unknown";
}

static void decode_with_llama_cpp(const std::string& model_path, const Request& req) {
    // Integration handoff:
    // 1) Convert prompt to tokens with llama.cpp tokenizer
    // 2) Feed token batches to llama_decode()
    // 3) Stream generated tokens to caller
    //
    // For this self-contained demo we simulate decode latency.
    const auto decode_time = (req.lane == rais::Lane::Interactive)
        ? std::chrono::milliseconds(40)
        : std::chrono::milliseconds(120);

    std::this_thread::sleep_for(decode_time);
    std::printf("[lane=%s] model=%s request=%d prompt=\"%s\"\n",
                lane_name(req.lane), model_path.c_str(), req.id, req.prompt.c_str());
}

int main() {
    const std::string model_path = "models/llama-3.2-1b-instruct-q4.gguf";
    rais::Scheduler sched({.num_workers = 4});

    std::vector<Request> requests = {
        {1, "Summarize today's build errors.", rais::Lane::Bulk},
        {2, "Write a user-facing reply.", rais::Lane::Interactive},
        {3, "Generate changelog bullets.", rais::Lane::Background},
        {4, "What changed in this PR?", rais::Lane::Interactive},
    };

    std::vector<rais::TaskHandle> handles;
    handles.reserve(requests.size());

    for (const auto& req : requests) {
        handles.push_back(sched.submit([model_path, req]() {
            decode_with_llama_cpp(model_path, req);
        }, req.lane));
    }

    for (auto& handle : handles) {
        handle.wait();
    }

    sched.shutdown();
    std::puts("\nllama.cpp scheduling demo finished.");
    return 0;
}
