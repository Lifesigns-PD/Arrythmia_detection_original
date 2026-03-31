
package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
)

// --------------------------------------------------------------------------
// Result holds the output of the Pan-Tompkins detection pipeline.
// --------------------------------------------------------------------------

// Result contains detected R-peaks and all intermediate signals.
type Result struct {
	RPeaks     []int     // Final R-peak sample indices
	Filtered   []float64 // Bandpass-filtered signal
	Derivative []float64 // Five-point derivative signal
	Squared    []float64 // Squared signal
	Integrated []float64 // Moving-window integrated signal
	NumBeats   int       // Number of detected beats
	MeanHRBpm  float64   // Estimated mean heart rate in BPM
}

// ==========================================================================
//  Stage 1 — Bandpass Filter (5–15 Hz)
// ==========================================================================

// butterworthBandpass applies a second-order Butterworth bandpass filter
// using a forward-backward (filtfilt-equivalent) approach to achieve
// zero-phase distortion.
//
// Since Go has no scipy, this implements the filter from first principles:
//   - Design second-order Butterworth sections for LP and HP
//   - Apply cascaded biquad IIR filters (forward then reverse)
func butterworthBandpass(signal []float64, fs int, lowcut, highcut float64, order int) []float64 {
	// We cascade a high-pass at lowcut with a low-pass at highcut.
	// Each is a second-order Butterworth section applied via filtfilt.

	// Step 1: Low-pass at highcut
	lp := butterworthLowpass(signal, fs, highcut, order)

	// Step 2: High-pass at lowcut (applied to LP output)
	bp := butterworthHighpass(lp, fs, lowcut, order)

	return bp
}

// butterworthLowpass designs and applies a second-order Butterworth low-pass
// filter using the bilinear transform, then applies it forward-backward.
func butterworthLowpass(signal []float64, fs int, cutoff float64, order int) []float64 {
	// Pre-warp the cutoff frequency
	nyquist := float64(fs) / 2.0
	wc := math.Tan(math.Pi * cutoff / nyquist) // warped frequency

	// Second-order section coefficients (Butterworth)
	// Transfer function: H(s) = 1 / (s^2 + sqrt(2)*s + 1)
	k := wc * wc
	sqrt2 := math.Sqrt(2.0)
	norm := 1.0 / (1.0 + sqrt2*wc + k)

	b := [3]float64{k * norm, 2.0 * k * norm, k * norm}
	a := [3]float64{1.0, 2.0 * (k - 1.0) * norm, (1.0 - sqrt2*wc + k) * norm}

	// Apply order/2 times (order=2 means one pass of the biquad)
	result := make([]float64, len(signal))
	copy(result, signal)
	for i := 0; i < order/2; i++ {
		result = filtfilt(b, a, result)
	}
	if order%2 != 0 {
		result = filtfilt(b, a, result)
	}

	return result
}

// butterworthHighpass designs and applies a second-order Butterworth high-pass
// filter using the bilinear transform, then applies it forward-backward.
func butterworthHighpass(signal []float64, fs int, cutoff float64, order int) []float64 {
	nyquist := float64(fs) / 2.0
	wc := math.Tan(math.Pi * cutoff / nyquist)

	k := wc * wc
	sqrt2 := math.Sqrt(2.0)
	norm := 1.0 / (1.0 + sqrt2*wc + k)

	b := [3]float64{norm, -2.0 * norm, norm}
	a := [3]float64{1.0, 2.0 * (k - 1.0) * norm, (1.0 - sqrt2*wc + k) * norm}

	result := make([]float64, len(signal))
	copy(result, signal)
	for i := 0; i < order/2; i++ {
		result = filtfilt(b, a, result)
	}
	if order%2 != 0 {
		result = filtfilt(b, a, result)
	}

	return result
}

// filtfilt applies a second-order IIR filter forward and then backward
// to achieve zero-phase filtering (equivalent to scipy.signal.filtfilt).
func filtfilt(b, a [3]float64, signal []float64) []float64 {
	n := len(signal)
	if n == 0 {
		return signal
	}

	// Forward pass
	forward := iirFilter(b, a, signal)

	// Reverse the forward output
	reversed := make([]float64, n)
	for i := 0; i < n; i++ {
		reversed[i] = forward[n-1-i]
	}

	// Backward pass (filter the reversed signal)
	backward := iirFilter(b, a, reversed)

	// Reverse again to restore original order
	result := make([]float64, n)
	for i := 0; i < n; i++ {
		result[i] = backward[n-1-i]
	}

	return result
}

// iirFilter applies a direct-form II second-order IIR filter.
func iirFilter(b, a [3]float64, signal []float64) []float64 {
	n := len(signal)
	output := make([]float64, n)

	// State variables (direct form II transposed)
	var z1, z2 float64

	for i := 0; i < n; i++ {
		x := signal[i]
		y := b[0]*x + z1
		z1 = b[1]*x - a[1]*y + z2
		z2 = b[2]*x - a[2]*y
		output[i] = y
	}

	return output
}

// ==========================================================================
//  Stage 2 — Five-Point Derivative
// ==========================================================================

// derivative computes the five-point derivative as specified in the
// original Pan-Tompkins paper.
//
//	y(n) = (1/8) * [ -x(n-2) - 2x(n-1) + 2x(n+1) + x(n+2) ]
//
// Provides a good approximation of the first derivative while suppressing
// high-frequency noise better than a simple first difference.
func derivative(signal []float64) []float64 {
	n := len(signal)
	result := make([]float64, n) // edges remain zero-padded

	for i := 2; i < n-2; i++ {
		result[i] = (-signal[i-2] - 2.0*signal[i-1] +
			2.0*signal[i+1] + signal[i+2]) / 8.0
	}

	return result
}

// ==========================================================================
//  Stage 3 — Squaring
// ==========================================================================

// squaring performs point-by-point squaring.
// Makes all values positive and non-linearly amplifies large slopes
// (QRS complexes) relative to smaller slopes (P/T waves).
func squaring(signal []float64) []float64 {
	result := make([]float64, len(signal))
	for i, v := range signal {
		result[i] = v * v
	}
	return result
}

// ==========================================================================
//  Stage 4 — Moving-Window Integration
// ==========================================================================

// movingWindowIntegration smooths the squared signal into a single broad
// pulse per QRS complex. The original paper recommends ~150 ms window.
//
//   - Too narrow → multiple peaks per QRS (split detection)
//   - Too wide   → merges adjacent QRS complexes (missed beats)
func movingWindowIntegration(signal []float64, fs int, windowSec float64) []float64 {
	windowSize := int(windowSec * float64(fs))
	if windowSize < 1 {
		windowSize = 1
	}

	n := len(signal)
	result := make([]float64, n)

	// Compute initial sum for the first window position
	halfWin := windowSize / 2

	for i := 0; i < n; i++ {
		start := i - halfWin
		end := i + (windowSize - halfWin)

		if start < 0 {
			start = 0
		}
		if end > n {
			end = n
		}

		sum := 0.0
		for j := start; j < end; j++ {
			sum += signal[j]
		}
		result[i] = sum / float64(windowSize)
	}

	return result
}

// ==========================================================================
//  Stage 5 — Adaptive Thresholding with Search-Back
// ==========================================================================

// findPeaks finds local maxima in the signal with a minimum distance
// constraint between peaks (similar to scipy.signal.find_peaks).
func findPeaks(signal []float64, minDistance int) []int {
	n := len(signal)
	if n < 3 {
		return nil
	}

	// Find all local maxima
	var candidates []int
	for i := 1; i < n-1; i++ {
		if signal[i] > signal[i-1] && signal[i] >= signal[i+1] {
			candidates = append(candidates, i)
		}
	}

	if len(candidates) == 0 {
		return nil
	}

	// Enforce minimum distance: greedily keep peaks by amplitude
	type peakInfo struct {
		idx int
		val float64
	}
	peaks := make([]peakInfo, len(candidates))
	for i, idx := range candidates {
		peaks[i] = peakInfo{idx, signal[idx]}
	}

	// Sort by amplitude descending
	sort.Slice(peaks, func(i, j int) bool {
		return peaks[i].val > peaks[j].val
	})

	kept := make(map[int]bool)
	var result []int

	for _, p := range peaks {
		tooClose := false
		for existing := range kept {
			if abs(p.idx-existing) < minDistance {
				tooClose = true
				break
			}
		}
		if !tooClose {
			kept[p.idx] = true
			result = append(result, p.idx)
		}
	}

	// Sort by index (chronological order)
	sort.Ints(result)
	return result
}

// adaptiveThresholding implements dual adaptive thresholding on both the
// integrated signal and the bandpass-filtered signal, with a search-back
// mechanism for missed beats.
//
// The algorithm maintains two running estimates:
//   - SPKI / NPKI : signal / noise peak levels on the integrated waveform
//   - SPKF / NPKF : signal / noise peak levels on the filtered waveform
//
// Decision rules (from the original paper):
//  1. A candidate peak must exceed THRESHOLD_I1 on the integrated signal
//     AND THRESHOLD_F1 on the filtered signal to be classified as a QRS.
//  2. If no QRS is found within 166% of the average RR interval,
//     the algorithm searches back with lower thresholds to rescue a beat.
//  3. After each classification the running estimates are updated.
func adaptiveThresholding(integrated, originalFiltered []float64, fs int) []int {
	// Initial peak finding on the integrated signal
	minDistance := int(0.2 * float64(fs)) // 200 ms refractory period
	peaks := findPeaks(integrated, minDistance)

	if len(peaks) == 0 {
		return nil
	}

	// Initialise adaptive thresholds (training on first 2 seconds)
	trainingEnd := int(2.0 * float64(fs))
	if trainingEnd > len(integrated) {
		trainingEnd = len(integrated)
	}

	// Find training peaks
	var trainingPeaks []int
	for _, p := range peaks {
		if p < trainingEnd {
			trainingPeaks = append(trainingPeaks, p)
		}
	}

	var spki, spkf float64
	if len(trainingPeaks) > 0 {
		spki = integrated[trainingPeaks[0]]
		spkf = math.Abs(originalFiltered[trainingPeaks[0]])
		for _, p := range trainingPeaks {
			if integrated[p] > spki {
				spki = integrated[p]
			}
			if math.Abs(originalFiltered[p]) > spkf {
				spkf = math.Abs(originalFiltered[p])
			}
		}
	} else {
		spki = maxSlice(integrated[:trainingEnd])
		spkf = maxAbsSlice(originalFiltered[:trainingEnd])
	}

	npki := 0.0
	npkf := 0.0

	thresholdI1 := npki + 0.25*(spki-npki)
	thresholdI2 := 0.5 * thresholdI1
	thresholdF1 := npkf + 0.25*(spkf-npkf)
	thresholdF2 := 0.5 * thresholdF1

	// RR interval tracking
	rrAverage := fs // Start with 1-second assumption (~60 bpm)
	var rrHistory []int
	rrMissedLimit := 1.66

	// Classification loop
	var qrsPeaks []int

	for _, peakIdx := range peaks {
		peakValI := integrated[peakIdx]
		peakValF := math.Abs(originalFiltered[peakIdx])

		// Primary threshold test
		isQRS := false

		if peakValI > thresholdI1 && peakValF > thresholdF1 {
			if len(qrsPeaks) > 0 {
				timeSinceLast := peakIdx - qrsPeaks[len(qrsPeaks)-1]
				refractorySamples := int(0.2 * float64(fs))
				tWaveLimit := int(0.36 * float64(fs))

				if timeSinceLast < refractorySamples {
					// Inside refractory — treat as T-wave / noise
					isQRS = false
				} else if timeSinceLast < tWaveLimit {
					// Between 200-360 ms: possible T-wave check
					if peakValI < 0.5*integrated[qrsPeaks[len(qrsPeaks)-1]] {
						isQRS = false
					} else {
						isQRS = true
					}
				} else {
					isQRS = true
				}
			} else {
				isQRS = true
			}
		}

		if isQRS {
			qrsPeaks = append(qrsPeaks, peakIdx)

			// Update signal peak estimates
			spki = 0.125*peakValI + 0.875*spki
			spkf = 0.125*peakValF + 0.875*spkf

			// Update RR history
			if len(qrsPeaks) >= 2 {
				rr := qrsPeaks[len(qrsPeaks)-1] - qrsPeaks[len(qrsPeaks)-2]
				rrHistory = append(rrHistory, rr)
				if len(rrHistory) > 8 {
					rrHistory = rrHistory[len(rrHistory)-8:]
				}
				rrAverage = meanInt(rrHistory)
			}
		} else {
			// Update noise peak estimates
			npki = 0.125*peakValI + 0.875*npki
			npkf = 0.125*peakValF + 0.875*npkf
		}

		// Recalculate thresholds
		thresholdI1 = npki + 0.25*(spki-npki)
		thresholdI2 = 0.5 * thresholdI1
		thresholdF1 = npkf + 0.25*(spkf-npkf)
		thresholdF2 = 0.5 * thresholdF1

		// Search-back for missed beats
		if len(qrsPeaks) >= 2 {
			rrCurrent := qrsPeaks[len(qrsPeaks)-1] - qrsPeaks[len(qrsPeaks)-2]

			if float64(rrCurrent) > rrMissedLimit*float64(rrAverage) {
				searchStart := qrsPeaks[len(qrsPeaks)-2] + int(0.2*float64(fs))
				searchEnd := qrsPeaks[len(qrsPeaks)-1] - int(0.2*float64(fs))

				for _, cand := range peaks {
					if cand <= searchStart || cand >= searchEnd {
						continue
					}

					candValI := integrated[cand]
					candValF := math.Abs(originalFiltered[cand])

					if candValI > thresholdI2 && candValF > thresholdF2 {
						// Rescue this beat
						qrsPeaks = append(qrsPeaks, cand)

						// Update signal peak (search-back uses half weight)
						spki = 0.25*candValI + 0.75*spki
						spkf = 0.25*candValF + 0.75*spkf
						break // Only rescue one beat per gap
					}
				}

				// Re-sort after insertion
				sort.Ints(qrsPeaks)
			}
		}
	}

	// Suppress unused variable warnings
	_ = thresholdI2
	_ = thresholdF2

	return qrsPeaks
}

// ==========================================================================
//  Stage 6 — Peak Refinement
// ==========================================================================

// refinePeaks refines detected QRS locations back to the TRUE R-peak in
// the original (unfiltered or cleaned) ECG signal.
//
// The integrated signal introduces a group delay and smoothing, so the
// detected peak position may be offset from the actual R-peak. This step
// searches a small window (±50 ms) around each detection and picks the
// maximum amplitude in the original signal.
func refinePeaks(detectedPeaks []int, originalSignal []float64, fs int, searchWindowSec float64) []int {
	if len(detectedPeaks) == 0 {
		return nil
	}

	searchSamples := int(searchWindowSec * float64(fs))
	seen := make(map[int]bool)
	var refined []int

	for _, peak := range detectedPeaks {
		start := peak - searchSamples
		if start < 0 {
			start = 0
		}
		end := peak + searchSamples + 1
		if end > len(originalSignal) {
			end = len(originalSignal)
		}

		// Find local max in original signal
		localMax := start
		for j := start + 1; j < end; j++ {
			if originalSignal[j] > originalSignal[localMax] {
				localMax = j
			}
		}

		if !seen[localMax] {
			seen[localMax] = true
			refined = append(refined, localMax)
		}
	}

	sort.Ints(refined)
	return refined
}

// ==========================================================================
//  PUBLIC API — Main Entry Point
// ==========================================================================

// PanTompkinsDetect runs the full Pan-Tompkins QRS detection pipeline.
//
// Parameters:
//   - signal:              ECG signal (1-D slice). Can be raw or pre-cleaned.
//   - fs:                  Sampling frequency in Hz.
//   - bandpassLow:         Low cutoff for bandpass filter (default 5.0 Hz).
//   - bandpassHigh:        High cutoff for bandpass filter (default 15.0 Hz).
//   - integrationWindowSec: Moving-window width in seconds (default 0.150 s).
//   - refine:              If true, refine peaks to original signal maxima.
//
// Returns a Result struct with R-peaks, intermediate signals, and HR.
func PanTompkinsDetect(signal []float64, fs int, bandpassLow, bandpassHigh, integrationWindowSec float64, refine bool) Result {
	// --- Stage 1: Bandpass Filter ---
	filtered := butterworthBandpass(signal, fs, bandpassLow, bandpassHigh, 2)

	// --- Stage 2: Five-Point Derivative ---
	deriv := derivative(filtered)

	// --- Stage 3: Squaring ---
	sq := squaring(deriv)

	// --- Stage 4: Moving-Window Integration ---
	integrated := movingWindowIntegration(sq, fs, integrationWindowSec)

	// --- Stage 5: Adaptive Thresholding with Search-Back ---
	qrsIndices := adaptiveThresholding(integrated, filtered, fs)

	// --- Stage 6: Refine to true R-peak locations ---
	var rPeaks []int
	if refine && len(qrsIndices) > 0 {
		rPeaks = refinePeaks(qrsIndices, signal, fs, 0.050)
	} else {
		rPeaks = qrsIndices
	}

	// --- Compute summary metrics ---
	numBeats := len(rPeaks)
	meanHR := 0.0
	if numBeats >= 2 {
		totalRR := 0.0
		for i := 1; i < numBeats; i++ {
			totalRR += float64(rPeaks[i]-rPeaks[i-1]) / float64(fs)
		}
		meanRR := totalRR / float64(numBeats-1)
		if meanRR > 0 {
			meanHR = 60.0 / meanRR
		}
	}

	return Result{
		RPeaks:     rPeaks,
		Filtered:   filtered,
		Derivative: deriv,
		Squared:    sq,
		Integrated: integrated,
		NumBeats:   numBeats,
		MeanHRBpm:  meanHR,
	}
}

// DetectRPeaks is a convenience wrapper that returns ONLY the R-peak indices.
// Uses default parameters: bandpass 5-15 Hz, 150 ms integration window, with refinement.
func DetectRPeaks(signal []float64, fs int) []int {
	result := PanTompkinsDetect(signal, fs, 5.0, 15.0, 0.150, true)
	return result.RPeaks
}

// ==========================================================================
//  Helper functions
// ==========================================================================

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func maxSlice(s []float64) float64 {
	if len(s) == 0 {
		return 0
	}
	m := s[0]
	for _, v := range s[1:] {
		if v > m {
			m = v
		}
	}
	return m
}

func maxAbsSlice(s []float64) float64 {
	if len(s) == 0 {
		return 0
	}
	m := math.Abs(s[0])
	for _, v := range s[1:] {
		a := math.Abs(v)
		if a > m {
			m = a
		}
	}
	return m
}

func meanInt(s []int) int {
	if len(s) == 0 {
		return 0
	}
	sum := 0
	for _, v := range s {
		sum += v
	}
	return sum / len(s)
}

// ==========================================================================
//  Self-Test (run with: go run pan_tompkins.go)
// ==========================================================================

// SelfTest generates a synthetic ECG signal and runs the Pan-Tompkins
// algorithm on it to verify correctness.
func SelfTest() {
	fmt.Println("============================================================")
	fmt.Println("  Pan-Tompkins QRS Detection (Go) — Self-Test")
	fmt.Println("============================================================")

	fs := 125
	duration := 10.0
	numSamples := int(duration * float64(fs))

	// Time axis
	t := make([]float64, numSamples)
	for i := range t {
		t[i] = float64(i) / float64(fs)
	}

	// Simulate ~75 bpm (RR ≈ 0.8 s) with Gaussian QRS-like spikes
	rrInterval := 0.8
	ecg := make([]float64, numSamples)
	var truePeaks []int

	for beat := 0; beat < int(duration/rrInterval); beat++ {
		center := float64(beat)*rrInterval + 0.1
		if center >= duration {
			break
		}
		truePeaks = append(truePeaks, int(center*float64(fs)))

		qrsWidth := 0.04 // 40 ms
		sigma := qrsWidth / 3.0
		for i := 0; i < numSamples; i++ {
			dt := t[i] - center
			ecg[i] += 1.0 * math.Exp(-(dt*dt)/(2.0*sigma*sigma))
		}
	}

	// Add noise (deterministic seed for reproducibility)
	rng := rand.New(rand.NewSource(42))
	for i := range ecg {
		ecg[i] += 0.05 * rng.NormFloat64()
	}

	// Run Pan-Tompkins
	result := PanTompkinsDetect(ecg, fs, 5.0, 15.0, 0.150, true)

	fmt.Println()
	fmt.Printf("  Sampling Rate     : %d Hz\n", fs)
	fmt.Printf("  Signal Duration   : %.0f s\n", duration)
	fmt.Printf("  True Beats        : %d\n", len(truePeaks))
	fmt.Printf("  Detected Beats    : %d\n", result.NumBeats)
	fmt.Printf("  Mean HR           : %.1f bpm\n", result.MeanHRBpm)
	fmt.Printf("  R-peak Indices    : %v\n", result.RPeaks)
	fmt.Println()
	fmt.Println("  ✓ Self-test complete.")
	fmt.Println("============================================================")
}

func main() {
	SelfTest()
}
