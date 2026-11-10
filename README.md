# Project 1: Camera Calibration

In this project, we are characterizing an image processing system. This means we are determining the intrinsic properties of a specific camera and lens combination. For simplicity, I have locked down photography settings where possible. I expect to need to focus based on physically moving closer/further from the subject matter. Additionally, I expect photos to be under or over exposed in an effort to keep the apeture constant. I intend to maintain the same shutter speed and ISO through a shooting session, but will note if shutter speed and/or ISO need to change to adequitely capture required data for the math. All of this may change as new assignments ask to look at different aspects of the image processing system.

My late father was always into optics. His love for it started with astronomy and telescopes in his college years and moved onto photography in his retirement. He while he played with a variety of photographic styles, he was particularly into astro- and sunset photography and unique lighting affects therein. My brothers and I each inherited several of his cameras and lenses, but I haven't had much of a chance to learn or practice photography. Most of what I have learned has specifically been for camera phones as I generally have that camera on me all the time. However there are a handful of photographic approaches I would like to try that physically cannot be done on a camera with a lens less than 10 mm. This class and project have created a wonderful excuse for me to pick up a camera or two to build my own optical and digital image processing intuition.

## The Camera to Model

I inherited the following two cameras from my father, and the iPhone is generally in my pocket: a Sony α7S II with Rokinon 24 mm lens and an iPhone 16 Pro Max. I have used a variety of cameras over the years and often prefer travel and macro photography with the camera most easily at hand. I enjoy astrophotography when I can leave the metro area and the weather agrees with me. Most of my signal processing experience has been more time-oriented so this class will help me improve my understanding of spacial and multi-channel signal processing.

## Camera change

I originally planned to calibrate two cameras: a Sony α7S II with a Rokinon lens, and an iPhone 16 Pro Max using the Truly Simple Raw Camera app. Both changed:

- **Sony α7S II with Rokinon 24 mm lens: dropped.** The lens is fully manual and its firmware does not pass EXIF data (aperture, focal length, etc.) to the camera, so those values cannot be recorded for the comparison measurements.
- **iPhone 16 Pro Max: now the only camera.** Starting with the _Noise and Linear Light_ assignment, I capture raw DNG files with the camera inside the **Adobe Lightroom** app. The Truly Simple Raw Camera app did not give enough control over shutter speed, which that assignment requires.

Earlier assignments that used a different camera or app are noted in their own folders.

## Current camera: iPhone 16 Pro Max

I use the [iPhone 16 Pro Max](https://support.apple.com/en-us/121032) with the 1× lens and no flash.

From Apple, the 1× lens is called the [48MP Fusion lens](https://support.apple.com/en-us/121032):

> 48MP Fusion: 24 mm, ƒ/1.78 aperture, second-generation sensor-shift optical image stabilization, 100% Focus Pixels, support for super-high-resolution photos (24MP and 48MP). Also enables 12MP 2× Telephoto: 48 mm, ƒ/1.78 aperture, second-generation sensor-shift optical image stabilization, 100% Focus Pixels.

Additional details from [GSM Arena](https://www.gsmarena.com/apple_iphone_16_pro_max-review-2750p5.php) and [Phone Arena](https://www.phonearena.com/news/iphone-16-camera_id159634).

The photos come out at 12 MP, 24 mm, ƒ/1.78.

### Capture setup

| Setting             | Value                                                              |
| :------------------ | :----------------------------------------------------------------- |
| Capture app         | Adobe Lightroom 11.6.x (iOS), built-in camera in Pro mode          |
| File format         | DNG (raw), quad-pixel binned 12 MP from the 48 MP sensor           |
| Lens                | 1× main camera (24 mm equivalent), ƒ/1.78; other lenses taped over |
| Raw image size      | 4032 × 3024 pixels                                                 |
| Color filter array  | Bayer, BGGR                                                        |
| ISO used            | 64 (datasheet noise measurement), 320 (comparison)                 |
| Exposure range      | 1/10000 s to 1 s                                                   |
| Flash               | Off                                                                |
| Ambient temperature | about 20 °C (68 °F), phone not warm                                |

Every measurement starts from the raw Bayer mosaic with the black level subtracted. Nothing is demosaiced, white balanced, gamma corrected or resampled, because each of those steps would change the noise or the edge profiles before they are measured. Pixel centers are at integer coordinates, so a frame's center is ((w − 1)/2, (h − 1)/2).

**Note on the raw data:** Adobe Lightroom applies a fixed, sharpening-like filter within each color plane before writing the DNG. It can be disabled in the app but was not for these captures. It is visible in the noise measurement (neighboring pixels' noise is anti-correlated) and it raises the measured MTF, so both results describe this camera _as used through Lightroom_.

## Project 1 datasheet

All values come from [`results.json`](results.json). Each entry is a value ± one standard deviation, obtained as stated in the last column.

| Quantity                               | Value                                        | Uncertainty is                                   |
| :------------------------------------- | :------------------------------------------- | :----------------------------------------------- |
| Camera                                 | iPhone 16 Pro Max back camera 6.765mm f/1.78 | —                                                |
| Capture app                            | Adobe Lightroom 11.6.21 (iOS)                | —                                                |
| CFA pattern                            | BGGR                                         | —                                                |
| Raw size                               | 4032 × 3024 px                               | —                                                |
| Binned from a higher-resolution sensor | yes (48 MP → 12 MP, 2 × 2)                   | two independent checks agree                     |
| Focal length                           | 2946.2 ± 4.4 px                              | calibration fit sd (f_x, f_y fully correlated)   |
| Focal length                           | 7.13 ± 0.09 mm                               | fit sd and pixel-pitch uncertainty in quadrature |
| Principal point x                      | 2019.8 ± 1.8 px                              | calibration fit sd                               |
| Principal point y                      | 1512.7 ± 2.1 px                              | calibration fit sd                               |
| Distortion k1                          | 0.117 ± 0.002                                | calibration fit sd                               |
| Distortion k2                          | −0.313 ± 0.005                               | calibration fit sd                               |
| Distortion p1                          | −0.0008 ± 0.0002                             | calibration fit sd                               |
| Distortion p2                          | −0.0001 ± 0.0002                             | calibration fit sd                               |
| RMS reprojection error                 | 2.27 ± 0.08 px                               | jackknife over views                             |
| Calibration views used                 | 30                                           | —                                                |
| Checkerboard square (measured)         | 20.00 ± 0.04 mm                              | sd of five ruler runs                            |
| MTF50, center                          | 0.349 ± 0.005 cycles/green-plane sample      | sd over frames                                   |
| MTF50, corner mean                     | 0.379 ± 0.013 cycles/green-plane sample      | propagated from the four corners                 |
| MTF50, worst corner                    | 0.353 ± 0.012 cycles/green-plane sample      | that corner's sd                                 |
| Conversion gain K                      | 0.2532 ± 0.0013 DN/e⁻                        | weighted fit, calibrated on simulated sensors    |
| Read noise                             | 7.24 ± 0.09 e⁻                               | weighted fit                                     |
| Dark current                           | 0.17 ± 0.04 e⁻/s                             | fit standard error                               |
| ISO of the noise measurement           | 64                                           | —                                                |
| Black level (measured)                 | 527.45 ± 0.05 DN                             | spread across color channels                     |
| White level                            | 4095 DN                                      | —                                                |
| Vendor NoiseProfile present            | yes                                          | —                                                |
| Dynamic range                          | 10.93 ± 0.01 stops (65.8 dB)                 | from the fit's intercept                         |
| ΔE₀₀ before color correction           | 12.38 ± 0.07                                 | sample sd across three charts                    |
| ΔE₀₀ after color correction            | 3.47 ± 0.09                                  | sample sd across three charts                    |
| ΔE₀₀ worst patch after correction      | 7.15 ± 0.57                                  | sample sd across three charts                    |

Dynamic range criterion: DR = log₂(S_max / S_min) with S_max = white level − black level and S_min = the read noise (SNR = 1 counting read noise only), from a pair-difference photon transfer fit below 10% of the span. The exact SNR = 1 point, which includes the signal's own shot noise, gives 10.83 stops. ColorChecker reference set: "ColorChecker24 - After November 2014".

## Reproducing the results

```bash
uv sync
uv run python project1.py      # all five measurements; writes results.json and output/project1/
uv run pytest                  # the test suite; runs on simulated data, no images needed
```

Every default (image folders, measured square size, which slanted-edge series feed the datasheet, the expected camera) is a named constant at the top of [`project1.py`](project1.py); flags override them (`uv run python project1.py --help`). Before any step runs, every input folder is checked against the expected camera, so a capture from another lens cannot slip in. The raw images and the full outputs are shared separately (links in the write-up). Opt-in tests run against real images, for example `uv run pytest --slanted-edges <folder>`.

Input images can be found on Google Drive, but are _not_ publically accessible: [images/](https://drive.google.com/drive/folders/1H4djw0uX8hTXDoawyD-eeMZ1KRqfIBkp?usp=drive_link)

## The five measurements

### 1. Geometry: focal length, principal point and distortion

| Quantity                  | Value                                                          |
| :------------------------ | :------------------------------------------------------------- |
| Focal length              | 2946.2 ± 4.4 px = 7.13 ± 0.09 mm                               |
| Principal point           | (2019.8 ± 1.8, 1512.7 ± 2.1) px; frame center (2015.5, 1511.5) |
| Distortion k1, k2, p1, p2 | 0.117, −0.313, −0.0008, −0.0001 (k3 held at 0)                 |
| RMS reprojection error    | 2.27 ± 0.08 px over 30 views                                   |
| Checkerboard square       | 20.00 ± 0.04 mm                                                |

**Method.** A printed checkerboard was photographed at many tilts and positions, its 54 inner corners found in each frame with OpenCV's sector-based detector, and a pinhole camera with radial and tangential distortion fitted to all views at once (`cv.calibrateCameraExtended`, the method of the professor's notebook 02). The focal length in pixels is the mean of f_x and f_y; in millimeters it uses the pixel pitch implied by the file's own focal lengths (2.420 µm). The RMS reprojection error says how far, on average, the fitted model puts each corner from where it was detected. Its uncertainty is a jackknife: the calibration is repeated with each view left out, and the spread of the results shows how much the number depends on which views were shot.

**Checks.** The whole pipeline recovers a known synthetic camera within its own reported uncertainties, and the pitch it uses (2.420 µm) agrees with the published 2.44 µm to 0.8%. These views were taken close to the board (170–200 mm), where the depth of field is only about 13 mm: that is why the RMS is above 2 px, and why the focal length reads 5.4% above the EXIF value (the lens is focused close). A later, sharper set taken from farther away turned out to come from the ultra-wide camera (see below), so it cannot stand in for the main camera, and these close-range values remain the datasheet's geometry until the checkerboard is reshot.

> **Add figure:** `output/project1/geometry/geometry_undistort.png` (a straight edge before and after undistortion) and one corner overlay from `output/project1/geometry/corners/`.

### 2. Spatial resolution: MTF

| Quantity            | Value (cycles per green-plane sample; green Nyquist 0.5) |
| :------------------ | :------------------------------------------------------- |
| MTF50, center       | 0.349 ± 0.005 (70% of green Nyquist)                     |
| MTF50, corner mean  | 0.379 ± 0.013                                            |
| MTF50, worst corner | 0.353 ± 0.012                                            |

**Method.** The MTF describes how much contrast survives at increasing spatial frequencies; MTF50 is the frequency where half is left. It is measured with the slanted-edge method (notebook 04): a straight edge tilted a few degrees is located row by row, the samples are binned by their distance from the edge into a four-times-oversampled edge profile, which is differentiated into the line spread function, whose Fourier transform is the MTF. A printed target with a center square and four corner squares gives both edge directions at the center and near each frame corner, from one green plane of the raw mosaic, at 501 mm. The center uncertainty is the spread over frames of one target position; a corner measured in two positions at the same distance from the frame center also counts the disagreement between them.

**Checks.** On simulated targets with a known blur, every region and both edge directions recover the expected MTF50 within 4%, and a simulated double image puts an MTF zero exactly where theory predicts. Photographing the center at two distances showed that at 250 mm the print, not the camera, limited the result (MTF50 0.207 vs 0.369 at 501 mm), so the datasheet uses 501 mm. The corners measure _higher_ than the center: the right side of the frame was sharper on two days with the phone, tripod, mount and location all changed, which points to the camera rather than the setup.

> **Add figure:** the mean MTF curves of the center series (`output/project1/mtf/mtf_center_501mm.png`), and an ESF/LSF/MTF panel for one edge.

### 3. Noise

| Quantity               | Value                                        |
| :--------------------- | :------------------------------------------- |
| Conversion gain K      | 0.2532 ± 0.0013 DN/e⁻ (ISO 64)               |
| Read noise             | 7.24 ± 0.09 e⁻                               |
| Dark current           | 0.17 ± 0.04 e⁻/s                             |
| Black level (measured) | 527.45 ± 0.05 DN (declared in the file: 528) |
| White level            | 4095 DN                                      |

**Method.** Pairs of identical flat-field frames over a range of exposures give, from their difference, the frame-to-frame noise at each signal level, with every fixed pattern cancelled. For a Poisson-Gaussian sensor the variance grows linearly with the signal: the slope is the conversion gain K and the intercept the read noise (notebook 05's photon transfer method). The line is fitted below 10% of the span, weighted by how precisely each pair's variance is known; the uncertainties come from that fit. The black level is measured from lens-capped frames at several exposure times (their level extrapolated to zero exposure), not taken from the file's declared value. Read noise and shot noise are equal at 52 e⁻ (13 DN above the black level); below that the camera is read-noise limited.

**Checks.** Two independent measurements agree: the photon transfer fit and the frame-to-frame noise of the dark frames both give 7.24 e⁻. A single frame, instead of a pair, would overstate K by 9.3%, because it counts the fixed pixel-to-pixel pattern as noise. The weighted fit's reported uncertainty matched the actual scatter of K over 30 simulated sensors.

The camera's own noise model in the DNG (`NoiseProfile`) gives larger values, by a factor that depends on ISO; the cause is an open question:

| From the DNG `NoiseProfile` | Unit  | ISO 64                  | ISO 320                 |
| :-------------------------- | :---- | :---------------------- | :---------------------- |
| Conversion gain, K          | DN/e⁻ | 0.3160 (1.25× measured) | 1.4004 (1.11× measured) |
| Read noise                  | e⁻    | 11.22 (1.52× measured)  | 2.68 (1.09× measured)   |

> **Add figure:** the photon transfer curve with the measured and declared black levels (`output/project1/noise/noise_ptc_black_level_comparison.png`) and the noise power spectrum (`noise_power_spectrum.png`), which shows Lightroom's filtering.

### 4. Dynamic range

| Quantity                                     | Value                        |
| :------------------------------------------- | :--------------------------- |
| Dynamic range                                | 10.93 ± 0.01 stops = 65.8 dB |
| Saturation capacity (derived: span / K)      | about 14,090 ± 70 e⁻         |
| Floor at exact SNR = 1 (shot noise included) | 10.83 stops                  |

**Method.** The dynamic range is the ratio of the largest signal (saturation level minus black level) to the smallest usable one, here the read noise (SNR = 1). Both come from the same photon transfer fit, so the conversion gain cancels: DR = log₂(span / √c), with c the fit's intercept. Its uncertainty is the intercept's uncertainty carried through. A stricter photographic floor of SNR = 10 lies above the noise knee and gives about 4.3 stops less.

**Checks.** On a simulated sensor with known noise the method returns 11.12 ± 0.03 stops against a true 11.10. The value matches the arithmetic the professor gave for the HW3 data.

> **Add figure:** the SNR curve with the knee and both floors marked (`output/project1/noise/noise_snr.png`).

### 5. Color accuracy

| Quantity                                   | Value        |
| :----------------------------------------- | :----------- |
| ΔE₀₀ before color correction               | 12.38 ± 0.07 |
| ΔE₀₀ after correction (mean of 24 patches) | 3.47 ± 0.09  |
| ΔE₀₀ worst patch after correction          | 7.15 ± 0.57  |

**Method.** Three ColorChecker charts were photographed under an LED light. Each patch's raw color is white balanced on a mid gray, scaled to the reference brightness by a fit over all six gray patches, and compared with the published values in CIEDE2000 (ΔE₀₀, where about 1 is a just-noticeable difference). One 3 × 3 color correction matrix fitted over all three charts, the matrix a camera's image processor would store, maps camera color to reference color (notebook 06). The uncertainty is the sample standard deviation across the three charts, so it covers what changes between captures but not effects shared by all of them, such as stray light (flare) in this setup.

**Checks.** A simulated camera with an exact linear response corrects to ΔE₀₀ below 10⁻⁶, and a white patch reading 12% low is detected. Setting brightness from the white patch alone, as in HW4, gave 4.10 ± 0.52; the six-gray fit removed most of the chart-to-chart spread, which came from the white patch reading differently on each chart.

> **Add figure:** ΔE₀₀ per patch before and after correction (`output/project1/color/color_led_dE_before_after.png`).

## Ambition: the ultra-wide camera

A checkerboard set taken (by accident) with the ultra-wide camera calibrated cleanly: RMS 0.97 px over 53 views, focal length about 1496 px, and about −10% barrel distortion at the outermost corner, distortion you can see by eye.

> **Add figure:** the ultra-wide's undistortion figure next to the main camera's.

## Next steps

- Reshoot the flat fields, dark frames and slanted edges with Lightroom's processing disabled, and confirm the noise spectrum goes flat; then remeasure K, the read noise and the MTF without it.
- **First:** reshoot the checkerboard target from a greater distance (0.6–0.8 m, where a tilted board stays in focus) with the correct camera, the 1× main camera with the other lenses taped over, and recalibrate. The current geometry comes from a close-range set, and the sharper long-distance set was taken with the ultra-wide by mistake.
- Rotate the phone 180° in its mount and repeat a corner MTF series: if the sharper side moves with the phone, it is the camera; if it stays with the scene, it is the setup.
- Photograph the MTF center series at about 1 m to see whether 501 mm is fully free of the print's blur.
- Measure flat fields at more ISOs from a steadier light source to explain the vendor `NoiseProfile` gap and its ISO dependence.
- Correct lens shading before sampling the color patches, and repeat under a high-CRI light.

## Former plan: Sony α7S II with Rokinon 24 mm F/1.4 lens (dropped)

For the early homework assignments and Project 1 reports I had planned to use a [Sony $\alpha$ 7s II camera](https://www.sony.com/lr/electronics/interchangeable-lens-cameras/ilce-7sm2) with a [Samyang / Rokinon 24mm F/1.4 ED AS IF UMC lens](https://rokinon.com/products/24mm-f1-4-full-frame-wide-angle). As much as possible, I had planned to keep it at the following settings:

| Setting       | Value              |
| :------------ | :----------------- |
| Focal Lenth   | 24 mm              |
| Apature       | 5.6                |
| Shutter Speed | Starting at 1/80\* |
| ISO           | 1600               |
| Flash         | Off                |

\* First setting modified for exposure.

However, statistical analysis benefits from comparison with recorded EXIF data to check models, math, and input data for accuracy. Since the Rokinon cannot pass EXIF data to the Sony body, I cannot gather the required EXIF data for this project and had to drop this camera/lens combination.

_Note:_ Large chunks of this README and the code were written by Claude Sonnet 5.5 and Opus 5.5 at my regular direction and review.
