#include <rais/scheduler.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct ServerConfig {
    std::string model = "llama";
    size_t workers = 4;
};

static void print_usage() {
    std::puts("Usage: ./rais_server [--model <name>] [--workers <n>]");
    std::puts("Input mode:");
    std::puts("  - normal line      -> Interactive lane");
    std::puts("  - bulk:<prompt>    -> Bulk lane");
    std::puts("  - quit             -> shutdown");
}

static ServerConfig parse_args(int argc, char** argv) {
    ServerConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage();
            std::exit(0);
        } else if (arg == "--model" && i + 1 < argc) {
            cfg.model = argv[++i];
        } else if (arg == "--workers" && i + 1 < argc) {
            cfg.workers = std::strtoul(argv[++i], nullptr, 10);
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_usage();
            std::exit(1);
        }
    }
    return cfg;
}

static const char* lane_name(rais::Lane lane) {
    return lane == rais::Lane::Interactive ? "Interactive" : "Bulk";
}

int main(int argc, char** argv) {
    const ServerConfig cfg = parse_args(argc, argv);
    rais::Scheduler sched({.num_workers = cfg.workers});

    std::mutex out_mu;
    std::vector<rais::TaskHandle> in_flight;
    int next_request_id = 1;

    {
        std::lock_guard<std::mutex> lock(out_mu);
        std::printf("RAIS server mode demo\n");
        std::printf("model=%s workers=%zu\n", cfg.model.c_str(), cfg.workers);
        std::puts("Enter prompts (type `quit` to exit).");
    }

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }
        if (line == "quit" || line == "exit") {
            break;
        }

        rais::Lane lane = rais::Lane::Interactive;
        std::string prompt = line;
        if (line.rfind("bulk:", 0) == 0) {
            lane = rais::Lane::Bulk;
            prompt = line.substr(5);
            if (!prompt.empty() && prompt.front() == ' ') {
                prompt.erase(prompt.begin());
            }
        }

        const int request_id = next_request_id++;
        {
            std::lock_guard<std::mutex> lock(out_mu);
            std::printf("[queued] id=%d lane=%s prompt=\"%s\"\n",
                        request_id, lane_name(lane), prompt.c_str());
        }

        in_flight.push_back(sched.submit([request_id, prompt, lane, cfg, &out_mu]() {
            const auto decode_time = (lane == rais::Lane::Interactive)
                ? std::chrono::milliseconds(60)
                : std::chrono::milliseconds(220);
            std::this_thread::sleep_for(decode_time);

            std::lock_guard<std::mutex> lock(out_mu);
            std::printf("[done ] id=%d lane=%s model=%s response=\"ok: %s\"\n",
                        request_id, lane_name(lane), cfg.model.c_str(), prompt.c_str());
        }, lane));
    }

    for (auto& h : in_flight) {
        h.wait();
    }
    sched.shutdown();

    std::puts("Server demo shutdown complete.");
    return 0;
}
