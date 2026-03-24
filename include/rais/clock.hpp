#pragma once

#include <cstdint>

#ifdef __APPLE__
#include <mach/mach_time.h>
#else
#include <ctime>
#endif

namespace rais {

/// High-resolution monotonic clock returning nanoseconds.
///
/// On macOS/Apple Silicon, uses mach_absolute_time() which is lower overhead
/// and higher resolution than clock_gettime.
inline uint64_t clock_ns() {
#ifdef __APPLE__
    // mach_absolute_time() returns ticks in a hardware-dependent unit.
    // mach_timebase_info gives the conversion factor: ns = ticks * numer / denom.
    // On Apple Silicon, numer == denom == 1, so ticks ARE nanoseconds.
    // We cache the timebase info in a static to avoid repeated syscalls.
    static const auto info = []() {
        mach_timebase_info_data_t i;
        mach_timebase_info(&i);
        return i;
    }();
    uint64_t ticks = mach_absolute_time();
    return ticks * info.numer / info.denom;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL
         + static_cast<uint64_t>(ts.tv_nsec);
#endif
}

} // namespace rais
