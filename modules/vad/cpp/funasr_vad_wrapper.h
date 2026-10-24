// funasr_vad_wrapper.h — C API for FSMN-VAD, so it can be called from
// Python (ctypes), Java (JNA), C#, or any other FFI language.
// Thin wrapper around funasr_vad_segments() from funasr_vad.h.
//
// Threading: each call opens/loads the GGUF model itself, so it is safe to
// call concurrently from multiple threads (each call has its own context).
#ifndef FUNASR_VAD_WRAPPER_H
#define FUNASR_VAD_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

// Run FSMN-VAD on a 16k mono float waveform.
//
//   gguf_path      path to fsmn-vad.gguf model file
//   wav_16k        input waveform, 16000 Hz mono, normalized to [-1, 1]
//   nsamples       number of samples in wav_16k
//   max_seg_ms     cap a single segment length in ms (<=0 -> 60000)
//   max_end_silence_ms  fixed trailing-silence cut threshold in ms
//                  (<=0 -> DEFAULT_SILENCE_SCHEDULE adaptive behavior)
//   segs_ms_out    caller buffer, filled with [start_ms, end_ms] pairs
//                  (2 ints per segment, interleaved)
//   out_capacity   number of (start,end) PAIRS segs_ms_out can hold
//   nthreads       compute threads (<=0 -> 8)
//
// Returns:
//   >= 0  number of segments found (may exceed out_capacity; only the first
//         out_capacity pairs are written)
//   -1    model load / compute error
//   -2    invalid arguments
int funasr_vad(const char* gguf_path, const float* wav_16k, int nsamples,
               int max_seg_ms, int max_end_silence_ms,
               int* segs_ms_out, int out_capacity, int nthreads);

#ifdef __cplusplus
}
#endif

#endif // FUNASR_VAD_WRAPPER_H
