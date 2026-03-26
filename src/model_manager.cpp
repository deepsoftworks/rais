#include <rais/model_manager.hpp>

namespace rais {

ModelManager::ModelManager(Scheduler& scheduler)
    : scheduler_(scheduler) {}

TaskHandle ModelManager::swap(const std::filesystem::path& new_model_path,
                              ModelLoadFn load_fn,
                              ModelUnloadFn unload_current_fn,
                              std::function<void()> on_ready) {
    // If a swap is already in progress, cancel it and take over.
    bool expected = false;
    if (!swap_in_progress_.compare_exchange_strong(expected, true,
            std::memory_order_acq_rel, std::memory_order_relaxed)) {
        // Another swap is running — cancel it
        current_swap_.cancel();
        // Wait for it to finish so we don't race on unload
        current_swap_.wait();
        // swap_in_progress_ is still true — we own it now
    }

    auto handle = scheduler_.submit([this, new_model_path, load_fn,
                                     unload_current_fn, on_ready]() {
        bool ok = load_fn(new_model_path);
        if (ok) {
            unload_current_fn();
            on_ready();
        }
        // release: publishes the flag reset so subsequent swap() calls
        // see it via acquire in compare_exchange_strong above.
        swap_in_progress_.store(false, std::memory_order_release);
    }, Lane::Background);

    current_swap_ = handle;
    return handle;
}

bool ModelManager::swap_in_progress() const {
    return swap_in_progress_.load(std::memory_order_acquire);
}

} // namespace rais
