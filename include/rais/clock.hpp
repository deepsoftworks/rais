#pragma once

#include <cstdint>
#include <mach/mach_time.h>

namespace rais {

/// High-resolution monotonic clock returning nanoseconds.
///
/// Uses mach_absolute_time() which is lower overhead and higher resolution
/// than clock_gettime. On Apple Silicon, numer == denom == 1, so ticks
/// are nanoseconds directly.
inline uint64_t clock_ns() {
    static const auto info = []() {
        mach_timebase_info_data_t i;
        mach_timebase_info(&i);
        return i;
    }();
    uint64_t ticks = mach_absolute_time();
    return ticks * info.numer / info.denom;
}

} // namespace rais
