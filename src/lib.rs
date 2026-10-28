use realfft::{RealFftPlanner, RealToComplex};
#[cfg(test)]
use rustfft::num_complex::Complex;
use rustfft::num_complex::Complex32;
use std::cell::RefCell;
use std::sync::Arc;
use wasm_bindgen::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq)]
struct AnalyzerKey {
    sample_rate: u32,
    n_fft: usize,
    win_length: usize,
    f_min: u32,
    f_max: u32,
    n_mels: usize,
}

impl AnalyzerKey {
    fn new(
        sample_rate: f32,
        n_fft: usize,
        win_length: usize,
        f_min: f32,
        f_max: f32,
        n_mels: usize,
    ) -> Self {
        Self {
            sample_rate: sample_rate.to_bits(),
            n_fft,
            win_length,
            f_min: f_min.to_bits(),
            f_max: f_max.to_bits(),
            n_mels,
        }
    }
}

thread_local! {
    static ANALYZER_CACHE: RefCell<Vec<(AnalyzerKey, MelSpectrogramAnalyzer)>> =
        RefCell::new(Vec::with_capacity(2));
}

#[wasm_bindgen]
pub struct MelSpectrogramAnalyzer {
    win_length: usize,
    window: Vec<f32>,
    // Each mel triangle covers one contiguous run of FFT bins, starting at
    // `mel_starts[mel]`, with its weights at
    // `mel_weights[mel_offsets[mel]..mel_offsets[mel + 1]]`. Summing a run is
    // a dot product, which vectorises; scattering bins into bands does not.
    mel_starts: Vec<usize>,
    mel_offsets: Vec<usize>,
    mel_weights: Vec<f32>,
    n_mels: usize,
    fft: Arc<dyn RealToComplex<f32>>,
    fft_input: Vec<f32>,
    fft_output: Vec<Complex32>,
    fft_scratch: Vec<Complex32>,
    magnitudes: Vec<f32>,
}

/// `weights · values` over four independent sums, so the loop vectorises.
fn dot(weights: &[f32], values: &[f32]) -> f32 {
    let mut sums = [0.0f32; 4];
    let weight_chunks = weights.chunks_exact(4);
    let value_chunks = values.chunks_exact(4);
    let tail: f32 = weight_chunks
        .remainder()
        .iter()
        .zip(value_chunks.remainder())
        .map(|(weight, value)| weight * value)
        .sum();
    for (weight, value) in weight_chunks.zip(value_chunks) {
        for lane in 0..4 {
            sums[lane] += weight[lane] * value[lane];
        }
    }
    (sums[0] + sums[1]) + (sums[2] + sums[3]) + tail
}

#[wasm_bindgen]
impl MelSpectrogramAnalyzer {
    #[wasm_bindgen(constructor)]
    pub fn new(
        sample_rate: f32,
        n_fft: usize,
        win_length: usize,
        f_min: f32,
        f_max: f32,
        n_mels: usize,
    ) -> Self {
        assert!(n_fft.is_power_of_two(), "n_fft must be a power of 2");
        assert!(n_fft >= win_length);
        assert!(win_length > 1);
        assert!(n_mels > 0);

        let n_freqs = n_fft / 2 + 1;
        let f_nyquist = sample_rate / 2.0;
        let m_min = hz_to_mel(f_min);
        let m_max = hz_to_mel(f_max);
        let f_points: Vec<f32> = (0..n_mels + 2)
            .map(|i| mel_to_hz(m_min + (m_max - m_min) * i as f32 / (n_mels + 1) as f32))
            .collect();
        let f_diff: Vec<f32> = f_points
            .windows(2)
            .map(|points| points[1] - points[0])
            .collect();
        let mut mel_starts = Vec::with_capacity(n_mels);
        let mut mel_offsets = Vec::with_capacity(n_mels + 1);
        let mut mel_weights = Vec::with_capacity(n_freqs * 2);
        for mel_index in 0..n_mels {
            mel_offsets.push(mel_weights.len());
            let mut start = None;
            for frequency_index in 0..n_freqs {
                let frequency = f_nyquist * frequency_index as f32 / (n_freqs - 1) as f32;
                let down_slope = (frequency - f_points[mel_index]) / f_diff[mel_index];
                let up_slope = (f_points[mel_index + 2] - frequency) / f_diff[mel_index + 1];
                let weight = down_slope.min(up_slope).max(0.0)
                    * (2.0 / (f_points[mel_index + 2] - f_points[mel_index]));
                if weight > 0.0 {
                    start.get_or_insert(frequency_index);
                    mel_weights.push(weight);
                } else if start.is_some() {
                    // A triangle is positive on one interval only.
                    break;
                }
            }
            mel_starts.push(start.unwrap_or(0));
        }
        mel_offsets.push(mel_weights.len());

        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);
        let fft_input = fft.make_input_vec();
        let fft_output = fft.make_output_vec();
        let fft_scratch = fft.make_scratch_vec();

        Self {
            win_length,
            window: hann_window(win_length),
            mel_starts,
            mel_offsets,
            mel_weights,
            n_mels,
            fft,
            fft_input,
            fft_output,
            fft_scratch,
            magnitudes: vec![0.0; n_freqs],
        }
    }

    pub fn compute_db(&mut self, waveform: &[f32], hop_length: usize, top_db: f32) -> Vec<f32> {
        assert!(self.win_length >= hop_length);
        assert!(hop_length > 0);

        let pad_size = (self.win_length - hop_length) / 2;
        let padded_length = waveform.len() + pad_size * 2;
        let frame_end = padded_length
            .checked_sub(self.win_length + pad_size)
            .expect("waveform is too short for the configured window and hop lengths");
        let frame_count = frame_end.div_ceil(hop_length);
        let mut result = Vec::with_capacity(frame_count * self.n_mels);

        for frame_start in (0..frame_end).step_by(hop_length) {
            // The FFT uses its input as scratch, so every sample is rewritten.
            let (windowed, zero_padding) = self.fft_input.split_at_mut(self.win_length);
            match frame_start.checked_sub(pad_size) {
                Some(first) if first + self.win_length <= waveform.len() => {
                    let samples = &waveform[first..first + self.win_length];
                    for ((output, sample), weight) in
                        windowed.iter_mut().zip(samples).zip(&self.window)
                    {
                        *output = sample * weight;
                    }
                }
                _ => {
                    // Frames overlapping the padding read zeros outside the waveform.
                    for (window_index, output) in windowed.iter_mut().enumerate() {
                        let padded_index = frame_start + window_index;
                        *output = if padded_index >= pad_size
                            && padded_index < pad_size + waveform.len()
                        {
                            waveform[padded_index - pad_size] * self.window[window_index]
                        } else {
                            0.0
                        };
                    }
                }
            }
            zero_padding.fill(0.0);
            self.fft
                .process_with_scratch(
                    &mut self.fft_input,
                    &mut self.fft_output,
                    &mut self.fft_scratch,
                )
                .expect("reusable real FFT buffers have inconsistent lengths");

            for (magnitude, value) in self.magnitudes.iter_mut().zip(&self.fft_output) {
                *magnitude = (value.re * value.re + value.im * value.im).sqrt();
            }
            for mel_index in 0..self.n_mels {
                let weights =
                    &self.mel_weights[self.mel_offsets[mel_index]..self.mel_offsets[mel_index + 1]];
                let start = self.mel_starts[mel_index];
                result.push(dot(weights, &self.magnitudes[start..start + weights.len()]));
            }
        }

        let mut max_db = f32::NEG_INFINITY;
        for value in result.iter_mut() {
            *value = 20.0 * value.max(1e-10).log10();
            max_db = max_db.max(*value);
        }
        for value in result.iter_mut() {
            *value = value.max(max_db - top_db);
        }
        result
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_cached(
    sample_rate: f32,
    waveform: &[f32],
    n_fft: usize,
    win_length: usize,
    hop_length: usize,
    f_min: f32,
    f_max: f32,
    n_mels: usize,
    top_db: f32,
) -> Vec<f32> {
    let key = AnalyzerKey::new(sample_rate, n_fft, win_length, f_min, f_max, n_mels);
    ANALYZER_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let analyzer_index = cache.iter().position(|(cached_key, _)| *cached_key == key);
        if let Some(analyzer_index) = analyzer_index {
            if analyzer_index + 1 < cache.len() {
                let analyzer = cache.remove(analyzer_index);
                cache.push(analyzer);
            }
        } else {
            if cache.len() == 2 {
                cache.remove(0);
            }
            cache.push((
                key,
                MelSpectrogramAnalyzer::new(sample_rate, n_fft, win_length, f_min, f_max, n_mels),
            ));
        }
        cache
            .last_mut()
            .expect("spectrogram analyzer cache must contain the requested configuration")
            .1
            .compute_db(waveform, hop_length, top_db)
    })
}

#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn mel_spectrogram_db(
    sample_rate: f32,
    waveform: Vec<f32>,
    n_fft: usize,
    win_length: usize,
    hop_length: usize,
    f_min: f32,
    f_max: f32,
    n_mels: usize,
    top_db: f32,
) -> JsValue {
    let flat = compute_cached(
        sample_rate,
        &waveform,
        n_fft,
        win_length,
        hop_length,
        f_min,
        f_max,
        n_mels,
        top_db,
    );
    let rows: Vec<&[f32]> = flat.chunks_exact(n_mels).collect();
    serde_wasm_bindgen::to_value(&rows).unwrap()
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn mel_spectrogram(
    sample_rate: f32,
    waveform: Vec<f32>,
    n_fft: usize,
    win_length: usize,
    hop_length: usize,
    f_min: f32,
    f_max: f32,
    n_mels: usize,
) -> Vec<Vec<f32>> {
    let spectrogram = spectrogram(waveform, n_fft, win_length, hop_length, true);
    let n_freqs = spectrogram[0].len();
    let fbanks = mel_filter_bank(n_freqs, f_min, f_max, n_mels, sample_rate);

    let mel_spec = spectrogram
        .iter()
        .map(|spec_row| {
            (0..n_mels)
                .map(|j| {
                    spec_row
                        .iter()
                        .zip(fbanks.iter())
                        .map(|(&s, fbank)| s * fbank[j])
                        .sum()
                })
                .collect()
        })
        .collect();

    mel_spec
}

#[cfg(test)]
fn spectrogram(
    waveform: Vec<f32>,
    n_fft: usize,
    win_length: usize,
    hop_length: usize,
    onesided: bool,
) -> Vec<Vec<f32>> {
    assert!(n_fft >= win_length);
    assert!(win_length >= hop_length);

    let pad_size = (win_length - hop_length) / 2;

    let padded_waveform: Vec<f32> = vec![0.0; pad_size]
        .into_iter()
        .chain(waveform)
        .chain(vec![0.0; pad_size])
        .collect();

    let mut spectrogram = Vec::new();
    for i in (0..padded_waveform.len() - win_length - pad_size).step_by(hop_length) {
        let window = &padded_waveform[i..i + win_length];
        let windowed_input: Vec<f32> = window
            .iter()
            .zip(hann_window(win_length).iter())
            .map(|(a, b)| a * b)
            .collect();
        let fft_output: Vec<Complex<f32>> = fft(windowed_input, n_fft).into_iter().collect();
        let magnitude_spectrum: Vec<f32> = fft_output.iter().map(|x| x.norm()).collect();
        spectrogram.push(magnitude_spectrum);
    }

    if onesided {
        spectrogram
            .iter()
            .map(|row| row.iter().take(n_fft / 2 + 1).cloned().collect())
            .collect()
    } else {
        spectrogram
    }
}

#[cfg(test)]
fn fft(input: Vec<f32>, n_fft: usize) -> Vec<Complex<f32>> {
    let num_samples = input.len();
    assert!(n_fft.is_power_of_two(), "n_fft must be a power of 2");
    assert!(
        num_samples <= n_fft,
        "n must be less than or equal to n_fft"
    );

    if num_samples <= 1 {
        return input.into_iter().map(|x| Complex::new(x, 0.0)).collect();
    }

    let padded_input = if num_samples < n_fft {
        let padding = vec![0.0; n_fft - num_samples];
        input.clone().into_iter().chain(padding).collect()
    } else {
        input.clone()
    };

    // Split into even and odd parts
    let even: Vec<f32> = padded_input.iter().step_by(2).cloned().collect();
    let odd: Vec<f32> = padded_input.iter().skip(1).step_by(2).cloned().collect();

    // Recursive FFT on even and odd parts
    let even_fft = fft(even, n_fft / 2);
    let odd_fft = fft(odd, n_fft / 2);

    // Combine results
    let mut output = vec![Complex::new(0.0, 0.0); n_fft];
    for k in 0..(n_fft / 2) {
        let t = odd_fft[k]
            * Complex::from_polar(1.0, -2.0 * std::f32::consts::PI * k as f32 / n_fft as f32);
        output[k] = even_fft[k] + t;
        output[k + n_fft / 2] = even_fft[k] - t; // Exploit symmetry
    }
    output
}

fn hann_window(length: usize) -> Vec<f32> {
    (0..length)
        .map(|n| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / (length - 1) as f32).cos()))
        .collect()
}

#[cfg(test)]
fn mel_filter_bank(
    n_freqs: usize,
    f_min: f32,
    f_max: f32,
    n_mels: usize,
    sample_rate: f32,
) -> Vec<Vec<f32>> {
    let f_nyquist = sample_rate / 2.0;

    let all_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| f_nyquist * i as f32 / (n_freqs - 1) as f32)
        .collect(); // (n_freqs,)

    let m_min = hz_to_mel(f_min);
    let m_max = hz_to_mel(f_max);

    let m_points: Vec<f32> = (0..n_mels + 2)
        .map(|i| m_min + (m_max - m_min) * i as f32 / (n_mels + 1) as f32)
        .collect(); // (n_mels + 2,)

    let f_points: Vec<f32> = m_points.iter().map(|&mel| mel_to_hz(mel)).collect();

    let f_diff: Vec<f32> = f_points
        .iter()
        .skip(1)
        .zip(f_points.iter().take(f_points.len() - 1))
        .map(|(f2, f1)| f2 - f1)
        .collect(); // (n_mels + 1,)

    let slopes: Vec<Vec<f32>> = all_freqs
        .iter()
        .map(|&f| f_points.iter().map(|&fp| fp - f).collect())
        .collect(); // (n_freqs, n_mels + 2)

    let down_slopes: Vec<Vec<f32>> = slopes
        .iter()
        .map(|slope_slice| {
            slope_slice
                .iter()
                .take(n_mels)
                .zip(f_diff.iter().take(n_mels))
                .map(|(slope, &diff)| -1.0 * slope / diff)
                .collect()
        })
        .collect(); // (n_freqs, n_mels)

    let up_slopes: Vec<Vec<f32>> = slopes
        .iter()
        .map(|slope_slice| {
            slope_slice
                .iter()
                .skip(2)
                .take(n_mels)
                .zip(f_diff.iter().skip(1).take(n_mels))
                .map(|(slope, &diff)| slope / diff)
                .collect()
        })
        .collect();

    let mut fbanks: Vec<Vec<f32>> = up_slopes
        .iter()
        .zip(down_slopes.iter())
        .map(|(up, down)| {
            let row = down
                .iter()
                .zip(up.iter())
                .map(|(&d, &u)| d.min(u).max(0.0)) // Use both up and down slopes
                .collect::<Vec<f32>>();
            row
        })
        .collect();

    // Apply Slaney normalization
    for i in 0..n_mels {
        let enorm = 2.0 / (f_points[i + 2] - f_points[i]);
        for fbank in fbanks.iter_mut() {
            fbank[i] *= enorm;
        }
    }

    fbanks
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

#[cfg(test)]
fn amplitude_to_db(amplitudes: Vec<Vec<f32>>, top_db: f32) -> Vec<Vec<f32>> {
    let dbs: Vec<Vec<f32>> = amplitudes
        .iter()
        .map(|row| {
            row.iter()
                .map(|&amp| 20.0 * amp.max(1e-10).log10())
                .collect()
        })
        .collect();

    let max_db = dbs
        .iter()
        .map(|row| row.iter().cloned().fold(f32::NEG_INFINITY, f32::max))
        .fold(f32::NEG_INFINITY, f32::max);

    dbs.iter()
        .map(|row| row.iter().map(|&amp| amp.max(max_db - top_db)).collect())
        .collect()
}

#[cfg(test)]
fn _assert_complex_eq(left: Complex<f32>, right: Complex<f32>) {
    // tolerance for floating-point comparison
    const EPSILON: f32 = 1e-5;

    assert!(
        (left.re - right.re).abs() < EPSILON,
        "left: {:?}, right: {:?}",
        left,
        right
    );
    assert!(
        (left.im - right.im).abs() < EPSILON,
        "left: {:?}, right: {:?}",
        left,
        right
    );
}

#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use rustfft::num_complex::Complex;

    fn test_waveform(length: usize, sample_rate: f32) -> Vec<f32> {
        (0..length)
            .map(|sample| {
                let time = sample as f32 / sample_rate;
                0.43 * (2.0 * std::f32::consts::PI * 440.0 * time).sin()
                    + 0.24 * (2.0 * std::f32::consts::PI * 7_123.0 * time).sin()
                    + 0.12 * (2.0 * std::f32::consts::PI * 21_337.0 * time).sin()
            })
            .collect()
    }

    fn assert_optimized_matches_legacy(
        n_fft: usize,
        win_length: usize,
        hop_length: usize,
        n_mels: usize,
        waveform: Vec<f32>,
    ) {
        let sample_rate = 48_000.0;
        let legacy: Vec<f32> = amplitude_to_db(
            mel_spectrogram(
                sample_rate,
                waveform.clone(),
                n_fft,
                win_length,
                hop_length,
                20.0,
                24_000.0,
                n_mels,
            ),
            90.0,
        )
        .into_iter()
        .flatten()
        .collect();
        let mut analyzer =
            MelSpectrogramAnalyzer::new(sample_rate, n_fft, win_length, 20.0, 24_000.0, n_mels);
        let optimized = analyzer.compute_db(&waveform, hop_length, 90.0);
        let cached = compute_cached(
            sample_rate,
            &waveform,
            n_fft,
            win_length,
            hop_length,
            20.0,
            24_000.0,
            n_mels,
            90.0,
        );

        assert_eq!(optimized.len(), legacy.len());
        assert_eq!(cached, optimized);
        let mut maximum_difference = 0.0_f32;
        let mut difference_sum = 0.0_f64;
        for (optimized_value, legacy_value) in optimized.iter().zip(legacy.iter()) {
            let difference = (optimized_value - legacy_value).abs();
            maximum_difference = maximum_difference.max(difference);
            difference_sum += difference as f64;
        }
        let mean_difference = difference_sum / legacy.len() as f64;
        assert!(
            maximum_difference <= 0.02,
            "maximum dB difference was {maximum_difference}; mean was {mean_difference}"
        );
        assert!(
            mean_difference <= 0.001,
            "mean dB difference was {mean_difference}; maximum was {maximum_difference}"
        );
    }

    #[test]
    fn test_fft_constant() {
        let input = vec![1.0, 0.0, 0.0, 0.0]; // Changed to real input
        let output = fft(input, 4);
        _assert_complex_eq(output[0], Complex::new(1.0, 0.0));
        _assert_complex_eq(output[1], Complex::new(1.0, 0.0));
        _assert_complex_eq(output[2], Complex::new(1.0, 0.0));
        _assert_complex_eq(output[3], Complex::new(1.0, 0.0));
    }

    #[test]
    fn test_fft_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0]; // Changed to real input
        let output = fft(input, 4);
        _assert_complex_eq(output[0], Complex::new(10.0, 0.0));
        _assert_complex_eq(output[1], Complex::new(-2.0, 2.0));
        _assert_complex_eq(output[2], Complex::new(-2.0, 0.0));
        _assert_complex_eq(output[3], Complex::new(-2.0, -2.0));
    }

    #[test]
    fn test_fft_with_length_eight() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]; // Changed to real input
        let output = fft(input, 8);
        _assert_complex_eq(output[0], Complex::new(10.0, 0.0));
        _assert_complex_eq(output[1], Complex::new(-0.41421356, -7.24264069));
        _assert_complex_eq(output[2], Complex::new(-2.0, 2.0));
        _assert_complex_eq(output[3], Complex::new(2.41421356, -1.24264069));
        _assert_complex_eq(output[4], Complex::new(-2.0, 0.0));
        _assert_complex_eq(output[5], Complex::new(2.41421356, 1.24264069));
        _assert_complex_eq(output[6], Complex::new(-2.0, -2.0));
        _assert_complex_eq(output[7], Complex::new(-0.41421356, 7.24264069));
    }

    #[test]
    fn optimized_analyzer_matches_legacy_at_2048() {
        for hop_length in [64, 512, 2048] {
            assert_optimized_matches_legacy(
                2048,
                2048,
                hop_length,
                256,
                test_waveform(2048 * 4, 48_000.0),
            );
        }
    }

    #[test]
    fn optimized_analyzer_matches_legacy_at_4096() {
        for hop_length in [128, 1024, 4096] {
            assert_optimized_matches_legacy(
                4096,
                4096,
                hop_length,
                512,
                test_waveform(4096 * 3, 48_000.0),
            );
        }
    }

    #[test]
    fn optimized_analyzer_matches_legacy_with_zero_padding() {
        // Neural Analog's displays pad each window to four times its length.
        for hop_length in [512, 2048] {
            assert_optimized_matches_legacy(
                8192,
                2048,
                hop_length,
                256,
                test_waveform(2048 * 6, 48_000.0),
            );
        }
    }

    #[test]
    fn optimized_analyzer_reuses_its_plan_across_overview_batches() {
        let waveform = test_waveform(2048 * 33, 48_000.0);
        let mut analyzer = MelSpectrogramAnalyzer::new(48_000.0, 2048, 2048, 20.0, 24_000.0, 256);
        let first = analyzer.compute_db(&waveform, 2048, 90.0);
        let second = analyzer.compute_db(&waveform, 2048, 90.0);
        assert_eq!(first.len(), 32 * 256);
        assert_eq!(second, first);
    }

    #[test]
    fn legacy_api_cache_is_bounded_to_two_lod_configurations() {
        ANALYZER_CACHE.with(|cache| cache.borrow_mut().clear());
        let waveform = test_waveform(4096 * 3, 48_000.0);
        for (n_fft, n_mels) in [(1024, 128), (2048, 256), (4096, 512)] {
            compute_cached(
                48_000.0, &waveform, n_fft, n_fft, n_fft, 20.0, 24_000.0, n_mels, 90.0,
            );
        }
        ANALYZER_CACHE.with(|cache| {
            let cache = cache.borrow();
            assert_eq!(cache.len(), 2);
            assert_eq!(cache[0].0.n_fft, 2048);
            assert_eq!(cache[1].0.n_fft, 4096);
        });
    }
}
