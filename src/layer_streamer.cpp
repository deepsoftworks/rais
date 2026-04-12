#include <rais/layer_streamer.hpp>

#include <cstdio>
#include <iomanip>
#include <sstream>

// Forward declare the Obj-C helper we need. MetalBufferPool returns
// id<MTLBuffer> as void*. To get the CPU-visible pointer we need
// [buffer contents], but we can't call Obj-C from a .cpp file.
// Instead, we rely on the fact that MTLStorageModeShared buffers
// have a stable CPU pointer that MetalBufferPool can expose.
// For now we pass the void* directly as both "buffer handle" and
// "writable pointer" — the pool uses MTLStorageModeShared so the
// void* returned by acquire() IS the MTLBuffer, and its .contents
// gives the CPU pointer. We'll get .contents via a thin helper.
//
// To avoid Obj-C in this file, we declare an extern helper that
// lives in metal_allocator.mm.
extern "C" void* rais_mtl_buffer_contents(void* buffer);

namespace rais {

LayerStreamer::LayerStreamer(Scheduler& scheduler,
                            MetalBufferPool& pool,
                            LayerStreamerConfig config)
    : scheduler_(scheduler)
    , pool_(pool)
    , config_(std::move(config))
    , slots_(config_.num_buffer_slots) {
    // Pre-acquire buffers from the pool for our ring
    for (auto& slot : slots_) {
        slot.buffer = pool_.acquire(config_.layer_size_bytes);
        if (slot.buffer) {
            slot.contents = rais_mtl_buffer_contents(slot.buffer);
        }
    }
}

LayerStreamer::~LayerStreamer() {
    cancel_all();
    for (auto& slot : slots_) {
        if (slot.buffer) {
            pool_.release(slot.buffer);
            slot.buffer = nullptr;
            slot.contents = nullptr;
        }
    }
}

std::filesystem::path LayerStreamer::layer_path(size_t idx) const {
    // Zero-padded 4-digit layer index: layer_0000, layer_0001, ...
    std::ostringstream name;
    name << "layer_" << std::setw(4) << std::setfill('0') << idx;
    return config_.model_dir / name.str();
}

void LayerStreamer::prefetch_layer(size_t layer_idx) {
    // mu_ must be held by caller
    if (layer_idx >= config_.num_layers) return;

    // Find a free slot (not occupied)
    for (auto& slot : slots_) {
        if (!slot.occupied && slot.buffer) {
            slot.occupied = true;
            slot.layer_idx = layer_idx;
            slot.read_handle = submit_file_read(
                scheduler_,
                layer_path(layer_idx),
                0,
                config_.layer_size_bytes,
                slot.contents);
            return;
        }
    }
    // No free slot — skip prefetch, will be fetched on demand
}

void LayerStreamer::start_prefetch(size_t start_layer) {
    std::lock_guard<std::mutex> lock(mu_);
    next_prefetch_ = start_layer;

    // Fill the pipeline by prefetching up to num_buffer_slots layers
    for (size_t i = 0; i < config_.num_buffer_slots &&
         (start_layer + i) < config_.num_layers; ++i) {
        prefetch_layer(start_layer + i);
    }
    next_prefetch_ = start_layer + config_.num_buffer_slots;
}

TaskHandle LayerStreamer::request_layer(size_t layer_idx,
                                        std::function<void(void* buffer)> on_ready) {
    std::lock_guard<std::mutex> lock(mu_);

    // Check if the layer is already in a slot (prefetched or in-flight)
    for (auto& slot : slots_) {
        if (slot.occupied && slot.layer_idx == layer_idx) {
            // Chain on_ready as a continuation of the read
            void* buf = slot.contents;
            return scheduler_.then(slot.read_handle, [on_ready, buf]() {
                on_ready(buf);
            }, Lane::Background);
        }
    }

    // Not prefetched — find a free slot and issue a read now
    for (auto& slot : slots_) {
        if (!slot.occupied && slot.buffer) {
            slot.occupied = true;
            slot.layer_idx = layer_idx;
            slot.read_handle = submit_file_read(
                scheduler_,
                layer_path(layer_idx),
                0,
                config_.layer_size_bytes,
                slot.contents);
            void* buf = slot.contents;
            return scheduler_.then(slot.read_handle, [on_ready, buf]() {
                on_ready(buf);
            }, Lane::Background);
        }
    }

    // All slots occupied — caller must release_layer first.
    // Return a no-op task that calls on_ready(nullptr) to signal failure.
    return scheduler_.submit([on_ready]() {
        on_ready(nullptr);
    }, Lane::Background);
}

void LayerStreamer::release_layer(size_t layer_idx) {
    std::lock_guard<std::mutex> lock(mu_);

    for (auto& slot : slots_) {
        if (slot.occupied && slot.layer_idx == layer_idx) {
            // Ensure the prior read task is fully complete before this slot is
            // recycled. This prevents the scheduler from still touching the
            // predecessor task/dependents while the handle is dropped.
            if (!slot.read_handle.done()) {
                slot.read_handle.wait();
            }
            slot.occupied = false;
            slot.layer_idx = SIZE_MAX;
            slot.read_handle = TaskHandle{};

            // Advance prefetch window: read the next layer into the freed slot
            if (next_prefetch_ < config_.num_layers) {
                prefetch_layer(next_prefetch_);
                ++next_prefetch_;
            }
            return;
        }
    }
}

void LayerStreamer::cancel_all() {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& slot : slots_) {
        if (slot.occupied) {
            slot.read_handle.cancel();
            slot.occupied = false;
            slot.layer_idx = SIZE_MAX;
            slot.read_handle = TaskHandle{};
        }
    }
}

} // namespace rais
