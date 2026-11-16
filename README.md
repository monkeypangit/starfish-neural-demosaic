# Starfish neural demosaic

Neural joint demosaicing and denoising for raw photos: PyTorch training for the
networks (X-Trans and Bayer), and `wgsl.py`, which compiles a trained model into
the WGSL shaders and `graph.txt` that run on the GPU through wgpu. The
[Starfish](https://github.com/monkeypangit/starfish) raw editor is built on it
and holds the model-agnostic executor (`wgpu-run/`) that loads the exports.

## Purpose
This project is intended to:

* Show the output quality of a basic neural joint demosaicing and denoising model
* Illustrate the viability of using wgpu as a distribution platform for neural models in raw photo editors
* Be a test bench for future demosaicing techniques
* Act as a way to connect with others interested in the topic

Ultimately the idea is to develop a solid pipeline for joint demosaicing + denoising + chromatic aberration correction that can be integrated into other projects.

## The demosaicing pipeline
The demosaicing is currently based on the [Deep Joint Demosaicking and Denoising by Gharbi et al. (2016)](https://groups.csail.mit.edu/graphics/demosaicnet/) paper and reference code. This project currently uses a slightly scaled down version of that model.

It differs from the DemosaicNet implementation in that it uses per CFA site filter banks in the full resolution output stage of the model. This allows the X-Trans version of the model to use the same packed trunk architecture as the Bayer version (packed as 6x6 instead 2x2). Also instead of doing 3x3 convolutions throughout the trunk the X-Trans model uses alternating 3x3 and 1x1 convolutions in an attempt to match the total receptive field and overall computational complexity of the Bayer version. The paper uses full scale trunk for X-Trans demosacing.

My understanding is that the DemosaicNet paper included the X-Trans model to show that their ideas generalized to other CFA layouts beyond Bayer. The X-Trans model was never intended to be optimal.

None of my changes have been proven to be better or preferable to the architecture used in the paper. My focus has been on making both models fit a common architectural pipeline and to match a certain performance target. The idea was to reach a performance target of 1-2 seconds to demosaic a 24 megapixel image on a 20-series Nvidia GPU, 2-5 seconds on modern iGPUs and 5-20 seconds on older iGPUs.

## Sample images
Here are some test images to show what the output looks like. The difference is difficult make out against the white page background. I also tried to pick fair comparisons to not oversell the difference. The images are sourced from [PIXLS.us](https://raw.pixls.us/). It was hard to find good images with a permissive license to illustrate the difference. You might want to download the full size images and pixel peep to really get an idea.

The differences are most easily spotted in high contrast areas such as dark tree leaves  against a bright sky, ripples on a water surface, bright objects against a dark background etc. It also shows in on smooth lightly textured regions such as the facade of a building or on rocks and pebbles. In such areas it shows up as subtle blotchy shifts in hue. Noisy regions also show a difference where the the grain becomes discolored and blotchy with many traditional methods.

You may want to download the full scale images and compare while zoomed in for a better view.

| | Starfish neural demosaic | Traditional hand crafted demosaic |
|---|---|---|
| Bayer | ![Bayer starfish](samples/bayer_starfish_crop.png) | ![Bayer other](samples/byer_hand_crafted_crop.png) |
| | [Full image](samples/bayer_starfish.png) | [Full image](samples/byer_hand_crafted.png) |
| X-Trans | ![X-Trans starfish](samples/xtrans_starfish_crop.png) | ![X-Trans other](samples/xtrans_hand_crafted_crop.png) |
| | [Full image](samples/xtrans_starfish.png) | [Full image](samples/xtrans_hand_crafted.png)

Also, please note that the neural demosaicing in this project is not without flaws. Most noticeably it suffers from luma zippering in both the Bayer and X-Trans model in high contrast and overexposed saturated areas. Hopefully this can be fixed with better training data sets or alterations to the model architecture.

## Known issues
- The neural demosaic is not perfect. Most notably there is an issue with discolorations around clipped higlights with sharp edges.

## Future improvements
The most important issue is to push up performance and output quality of the neural demosaicing. The Bayer output is currently looking quite good. Some moire remains but otherwise ok. X-Trans model shows luma zippering along edges. I am uncertain exactly why this happens but adding more layers to the full resolution head of the model and/or increasing the number of channels into the head layers help but makes the model slower.

- Fix issue with discolorations around clipped higlights
- Custom library of RAW files for model training collected from contributors or online sources of raw images with appropriate license.

## Wgpu as a distribution platform
This project also aims to show the viability of using wgpu/WebGPU as a relatively hassle free way to run neural models on a variety of hardware. The [Starfish](https://github.com/monkeypangit/starfish) app uses Rust ([iced](https://iced.rs) + [wgpu](https://wgpu.rs)). Iced is a cross platform UI library that uses wgpu for drawing the user interface. Wgpu is a cross platform Rust graphics API based on the WebGPU API. WebGPU in turn is a standardized API for interfacing with GPUs.

Wgpu wraps several platform specific libraries such as DirectX, Metal and Vulkan such that you can write the shader code once an run it on any system with minimal risk of technical issues. There are more purposeful alternatives such as [onnx](https://github.com/onnx/onnx). The alternatives can be more performant, but they can also be more complex to set up for the end user.

Ultimately it is hard to say what is the best setup. These things are still under active development and drivers and operating systems improve constantly. These projects explore one possible path using Pytorch for training and wgpu sharers for deployment.

## Training models
Training a model takes ~1 hour on my Nvidia RTX 5060 Ti 16GB. You can also use online serverless GPU services such as Modal as an alternative to running locally. See [AGENTS.md](AGENTS.md) for the technical details and commands.

The traning stage takes an image library such as the DIV2K and turns them into a large GPU buffer. During training the GPU draws random patches from this buffer, transforms them into linear camera space and applies random variations to them such as exposure compensation, flipping them, adding noise etc. This is intended to give the model under training the most varied set of data possible.

Initially I used hard patch mining similar to the DemosaicNet paper. The paper pre-processes the training images to find image patches with high frequency details. However, I opted out of this towards the end to simplify the training pipeline. It is convenient to create random patches from the test images on the fly. Maybe there is a way to identify hard patches during training and re-use them multiple times. However when adding noise to the training patches this will dominate what patches qualify as difficult rather than the contents of the original image.

The PyTorch code is in this repo; [AGENTS.md](AGENTS.md) has the layout, commands, model notes and training data. The DIV2K images are added by hand ([data/README.md](data/README.md)). The exports (`*_wgpu/`) are copied into the app repo's `models/` with `scripts/sync-models.sh` there. The bundled networks were trained on [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/): 8-bit sRGB images converted to linear light, with a simulated sensor. They are meant to be replaced by networks trained on real raw files with higher bit depth in the future.

## Background
I created this project after spending a little over a year researching hand crafted demosaicing methods such as RCD (for the Bayer CFA pattern) and Markesteijn (for the Fujifilm X-Trans CFA pattern). These are the methods traditionally used in open source raw image editors.

Methods like RCD, Markesteijn and others were created by very dedicated and academically skilled individuals. They are highly principled and produce great results for what they are. However, they were created in a time where computational resources were scarce and the cross platform support for GPU acceleration was not at the place it is today.

Initially I attempted to improve these traditional hand crafted methods. I had some success, but not to the point where I was happy with the output. The X-Trans CFA is an especially difficult case. Ultimately I decided to try nerual demosaicing. I found [Deep Joint Demosaicking and Denoising by Gharbi et al. (2016)](https://groups.csail.mit.edu/graphics/demosaicnet/) and I was blown away by the results. I had completely underestimated how much better these methods were in terms of output quality.

There are ways to improve the output quality of traditional methods like RCD and Markesteijn. Part of this project also includes writing a practical guide to demosaicing for people who are interested in the topic. The literature is dense and the sheer volume of published papers on the topic makes it difficult to approach.

Arguably we are currently in the period where hybrid hand crafted + nerual methods will serve most user the best from a quality/performance standpoint. However, I have not been able to find any such methods that seem mature enough to be implementable as is. So any such approach would include considerable effort in terms of innovation.

## License
The code is dual-licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option. Unless you explicitly state otherwise, any contribution you intentionally submit for inclusion in the work shall be dual-licensed as above, without any additional terms or conditions.

The weights in this repo (the `.pt` checkpoints and the `*_wgpu/` exports) were trained on the DIV2K dataset, which is provided for academic research. Its terms may apply to these weights. They will be replaced by weights trained on contributed data.
