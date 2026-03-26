#pragma once

#include <rais/scheduler.hpp>

#include <atomic>
#include <filesystem>
#include <functional>

namespace rais {

using ModelLoadFn   = std::function<bool(const std::filesystem::path& path)>;
using ModelUnloadFn = std::function<void()>;

/// Manages background model loading with zero-downtime swap.
///
/// The old model keeps serving while the new one loads on a Background task.
/// On success, the old model is unloaded and on_ready is called.
/// In-flight completions from the old model naturally expire via
/// generation_id checks in SpeculativeScheduler.
class ModelManager {
public:
    explicit ModelManager(Scheduler& scheduler);

    /// Load a new model in the background. On success: unload current,
    /// call on_ready. On failure: old model stays active, on_ready
    /// never fires. Returns a TaskHandle for the swap operation.
    TaskHandle swap(const std::filesystem::path& new_model_path,
                    ModelLoadFn load_fn,
                    ModelUnloadFn unload_current_fn,
                    std::function<void()> on_ready);

    /// True if a swap is currently in progress.
    bool swap_in_progress() const;

private:
    Scheduler& scheduler_;

    // Prevents concurrent swaps. Only one swap can be in-flight at a time.
    std::atomic<bool> swap_in_progress_{false};

    // Handle to the current swap task, used to cancel if a new swap is requested.
    TaskHandle current_swap_;
};

} // namespace rais
