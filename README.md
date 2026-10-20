# Project 1: Camera Calibration

In this project, we are characterizing an image processing system. This means we are determining the intrinsic properties of a specific camera and lens combination. For simplicity, I have locked down photography settings where possible. I expect to need to focus based on physically moving closer/further from the subject matter. Additionally, I expect photos to be under or over exposed in an effort to keep the apeture constant. I intend to maintain the same shutter speed and ISO through a shooting session, but will note if shutter speed and/or ISO need to change to adequitely capture required data for the math. All of this may change as new assignments ask to look at different aspects of the image processing system.

My late father was always into optics. His love for it started with astronomy and telescopes in his college years and moved onto photography in his retirement. He while he played with a variety of photographic styles, he was particularly into astro- and sunset photography and unique lighting affects therein. My brothers and I each inherited several of his cameras and lenses, but I haven't had much of a chance to learn or practice photography. Most of what I have learned has specifically been for camera phones as I generally have that camera on me all the time. However there are a handful of photographic approaches I would like to try that physically cannot be done on a camera with a lense less than 10 mm. This class and project have created a wonderful excuse for me to pick up a camera or two to build my own optical and digital image processing intuition.

## The Camera to Model

I inherited the following two cameras from my father, though the iPhone is generally in my pocket. I have used a variety of cameras over the years and often prefer travel and macro photography with the camera most easily at hand. I enjoy astrophotography when I can leave the metro area and the weather agrees with me. Most of my signal processing experience has been more time-oriented so this class will help me improve my understanding of spacial and multi-channel signal processing.

## Camera change

I originally planned to calibrate two cameras: a Sony α7S II with a Rokinon lens, and an iPhone 16 Pro Max using the Truly Simple Raw Camera app. Both changed:

- **Sony α7S II with Rokinon 24 mm lens: dropped.** The lens is fully manual and its firmware does not pass EXIF data (aperture, focal length, etc.) to the camera, so those values cannot be recorded for the comparison measurements.
- **iPhone 16 Pro Max: now the only camera.** Starting with the _Noise and Linear Light_ assignment, I capture raw DNG files with the camera inside the **Adobe Lightroom** app. The Truly Simple Raw Camera app did not give enough control over shutter speed, which that assignment requires.

Earlier assignments that used a different camera or app are noted in their own folders.

## Current camera: iPhone 16 Pro Max

I use the [iPhone 16 Pro Max](https://support.apple.com/en-us/121032) with the 1× lens and no flash.

From Apple, the 1× lens is called the [48MP Fusion lens](https://support.apple.com/en-us/121032):

> 48MP Fusion: 24 mm, ƒ/1.78 aperture, second-generation sensor-shift optical image stabilization, 100% Focus Pixels, support for super-high-resolution photos (24MP and 48MP). Also enables 12MP 2× Telephoto: 48 mm, ƒ/1.78 aperture, second-generation sensor-shift optical image stabilization, 100% Focus Pixels.

The photos come out at 12 MP, 24 mm, ƒ/1.78.

### Capture setup

| Setting             | Value                               |
| :------------------ | :---------------------------------- |
| Capture app         | Adobe Lightroom (camera, Pro mode)  |
| File format         | DNG (raw)                           |
| Lens                | 1× (24 mm equivalent), ƒ/1.78       |
| Raw image size      | 4032 × 3024 pixels (12 MP)          |
| Color filter array  | Bayer, BGGR                         |
| ISO used            | 64, 320                             |
| Exposure range      | 1/10000 s to 1 s                    |
| Flash               | Off                                 |
| Ambient temperature | about 20 °C (68 °F), phone not warm |

### Measured sensor and pipeline data

| Quantity                                  | Unit   | ISO 64          | ISO 320 |
| :---------------------------------------- | :----- | :-------------- | :------ |
| Black level stored in file (each channel) | DN     | 528             | 528     |
| White level stored in file                | DN     | 4095            | 4095    |
| Conversion gain, K                        | DN/e⁻  | 0.2567 ± 0.0019 | —       |
| Read noise, σᵣ                            | e⁻ RMS | 6.99 ± 0.32     | —       |
| Noise knee (σᵣ²)                          | e⁻     | 49              | —       |
| Full well (ADC clip)                      | e⁻     | ≈ 13,800        | —       |
| Dynamic range, log₂(full well / σᵣ)       | stops  | ≈ 10.9          | —       |
| Dark current (upper bound, 95%)           | e⁻/s   | < 0.3           | —       |
| Dark-frame mean, shortest exposure: R     | DN     | 527.43          | —       |
| Dark-frame mean, shortest exposure: G     | DN     | 527.50          | —       |
| Dark-frame mean, shortest exposure: B     | DN     | 527.40          | —       |
| Dark-frame mean, shortest exposure: G2    | DN     | 527.50          | —       |
| Dark-frame temporal variance (green)      | DN²    | 3.42            | —       |
| PRNU (fractional pixel-gain variation)    | %      | 0.7 to 0.9      | —       |

### Vendor noise model (DNG `NoiseProfile` tag)

Converted with W − B = 3567 DN: K = S·(W − B), σᵣ = √O / S.

| Quantity             | Unit   | ISO 64   | ISO 320  |
| :------------------- | :----- | :------- | :------- |
| S (normalized units) | —      | 8.858e-5 | 3.926e-4 |
| O (normalized units) | —      | 9.874e-7 | 1.111e-6 |
| Conversion gain, K   | DN/e⁻  | 0.3160   | 1.4004   |
| Read noise, σᵣ       | e⁻ RMS | 11.22    | 2.68     |

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
