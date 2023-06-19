
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

### Added

### Changed

### Fixed


## v0.2.0

### Added

* Introduce cherry.nn.Policy, cherry.nn.ActionValue, and cherry.nn.StateValue.
* Algorithm class utilities for: A2C, PPO, TRPO, DDPG, TD3, SAC, and DrQ/DrQv2.
* DMC examples for SAC, DrQ, and DrQv2.
* N-steps returns sampling in ExperienceReplay.

### Changed

* Discontinue most of cherry.wrappers.

### Fixed

* Fixes return value of StateNormalizer and RewardNormalizer wrappers.
* Requirements to generate docs.


## v0.1.4

### Fixed

* Support for torch 1.5 and new `_parse_to` behavior in ExperienceReplay. (thanks @ManifoldFR)


## v0.1.3

### Added

* A CHANGELOG.md file.

### Changed

* Travis testing with different versions of Python (3.6, 3.7), torch (1.1, 1.2, 1.3, 1.4), and torchvision (0.3, 0.4, 0.5).

### Fixed

* fix bug in `torch_wrapper` when use GPU by callling Tensor.cpu().detach().numpy() to convert CUDA tensor to numpy.(@walkacross)
* Bugfix when using `td.discount` with replays coming from vectorized environments (@galatolofederico) 
* env.action_size and env.state_size when the number of vectorized environments is 1. (thanks @galatolofederico)
* Actor-critic integration test being to finicky.
* `cherry.onehot` support for numpy's float and integer types. (thanks @ngoby)
