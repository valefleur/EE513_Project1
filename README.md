# Project 1 in EE513 Intro to Image Processing

In this project, we are characterizing an image processing system. This means we are determining the intrinsic properties of a specific camera and lens combination. For simplicity, I have locked down photography settings where possible. I expect to need to focus based on physically moving closer/further from the subject matter. Additionally, I expect photos to be under or over exposed in an effort to keep the apeture constant. I intend to maintain the same shutter speed and ISO through a shooting session, but will note if shutter speed and/or ISO need to change to adequitely capture required data for the math. All of this may change as new assignments ask to look at different aspects of the image processing system.

My late father was always into optics. His love for it started with astronomy and telescopes in his college years and moved onto photography in his retirement. He while he played with a variety of photographic styles, he was particularly into astro- and sunset photography and unique lighting affects therein. My brothers and I each inherited several of his cameras and lenses, but I haven't had much of a chance to learn or practice photography. Most of what I have learned has specifically been for camera phones as I generally have that camera on me all the time. However there are a handful of photographic approaches I would like to try that physically cannot be done on a camera with a lense less than 10 mm. This class and project have created a wonderful excuse for me to pick up a camera or two to build my own optical and digital image processing intuition.

## The Camera to Model

I inherited the following two cameras from my father, though the iPhone is generally in my pocket. I have used a variety of cameras over the years and often prefer travel and macro photography with the camera most easily at hand. I enjoy astrophotography when I can leave the metro area and the weather agrees with me. Most of my signal processing experience has been more time-oriented so this class will help me improve my understanding of spacial and multi-channel signal processing.

### Sony $\alpha$ 7s II with Rokinon 24mm F/1.4 lens

For homework assignments and Project 1 reports I plan to use a [Sony $\alpha$ 7s II camera](https://www.sony.com/lr/electronics/interchangeable-lens-cameras/ilce-7sm2) with a [Samyang / Rokinon 24mm F/1.4 ED AS IF UMC lens](https://rokinon.com/products/24mm-f1-4-full-frame-wide-angle). As much as possible, I plan to keep it at the following settings:

| Setting       | Value              |
| :------------ | :----------------- |
| Focal Lenth   | 24 mm              |
| Apature       | 5.6                |
| Shutter Speed | Starting at 1/80\* |
| ISO           | 1600               |
| Flash         | Off                |

\* First setting modified for exposure.

### iPhone 16 Pro Max

For kicks, I will also be running analysis and comparison using an [iPhone 16 Pro Max](https://support.apple.com/en-us/121032) using the [Truly Simple Raw Camera](https://trulysimpletools.com/trulysimplerawcamera/) app and the $1$x lens and no flash.
