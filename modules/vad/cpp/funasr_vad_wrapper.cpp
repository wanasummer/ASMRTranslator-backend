// funasr_vad_wrapper.cpp — C API implementation (see funasr_vad_wrapper.h).
#include "funasr_vad_wrapper.h"
#include "funasr_vad.h"

#include <utility>
#include <vector>

extern "C" {

int funasr_vad(const char* gguf_path, const float* wav_16k, int nsamples,
               int max_seg_ms, int max_end_silence_ms,
               int* segs_ms_out, int out_capacity, int nthreads) {
    if (!gguf_path || !wav_16k || nsamples <= 0 || !segs_ms_out) return -2;
    std::vector<float> wav(wav_16k, wav_16k + nsamples);

    std::vector<std::pair<int, int>> segs;
    bool ok = funasr_vad_segments(gguf_path, wav, max_seg_ms, max_end_silence_ms,
                                  segs, nthreads > 0 ? nthreads : 8);
    if (!ok) return -1;

    const int total = static_cast<int>(segs.size());
    const int n = total < out_capacity ? total : out_capacity;
    for (int i = 0; i < n; ++i) {
        segs_ms_out[i * 2]     = segs[i].first;
        segs_ms_out[i * 2 + 1] = segs[i].second;
    }
    return total;
}

} // extern "C"
