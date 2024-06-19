# xgb2code

`xgb2code` - generate code for an XGB model

## Description

This program takes an XGB model (in JSON format) and generates code for it.

Generating code for a model avoids having to call out to a different
language (e.g., C) as well as avoids the need for using the XGB libraries
at runtime.

## Model Support

Currently binary classification is supported.

## Supported Languages

Currently `xgb2code` supports generating Go code.

## Usage

```bash
$ ./xgb2code -h
Usage of ./xgb2code:
  -function-name string
        The function name to use. Must be a valid Go function name.
  -go-package-name string
        The package name to use when generating Go code. Must be a valid Go package name.
  -input-json string
        Path to the model as JSON
  -language string
        Language to generate code for. Currently 'go' is supported. (default "go")
  -output-file string
        The file to write to
```

## Example Usage

```bash
$ ./xgb2code -function-name predict \
             -go-package-name main
             -input-json testdata/small-model/model.json \
             -language go \
             -output-file predict.go
```

produces a file `predict.go` where the primary model prediction function
has the signature:

```go
func predict(data []*float32, predMargin bool) float32 {
```

## Installation

[Release binaries and
packages](https://github.com/maxmind/mmdbinspect/releases) have been made
available for several popular platforms. Simply download the binary for
your platform and run it.

## Bug Reports

Please report bugs by filing an issue with our GitHub issue tracker at
[https://github.com/maxmind/xgb2code/issues](https://github.com/maxmind/xgb2code/issues).

## Copyright and License

This software is Copyright (c) 2022 - 2024 by MaxMind, Inc.

This is free software, licensed under the [Apache License, Version 2.0](LICENSE-APACHE) or the [MIT License](LICENSE-MIT), at your option.
