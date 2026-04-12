#include <rais/scheduler.hpp>

#include <cstdio>

int main() {
    rais::Scheduler sched;

    auto handle = sched.submit([&]() {
        std::puts("generate(prompt)");
    }, rais::Lane::Interactive);

    handle.wait();
    sched.shutdown();
    return 0;
}
