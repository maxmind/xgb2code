// This program runs a command line program that generates Go code from an xgb model in JSON format.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/maxmind/xgb2code/gen"
)

type language string

const languageGo language = "go"

func main() {
	inputJSON := flag.String("input-json", "", "Path to the model as JSON")

	lang := flag.String(
		"language",
		string(languageGo),
		"Language to generate code for. Currently 'go' is supported.",
	)

	packageName := flag.String(
		"go-package-name",
		"",
		"The package name to use when generating Go code. Must be a valid Go package name.",
	)

	funcName := flag.String(
		"function-name",
		"",
		"The function name to use. Must be a valid Go function name.",
	)

	versionFlag := flag.Bool(
		"version",
		false,
		"Print version and exit.",
	)

	outputFile := flag.String("output-file", "", "The file to write to")

	flag.Parse()

	if *versionFlag {
		fmt.Printf("xgb2code v%s\n", version)
		os.Exit(0)
	}

	if *inputJSON == "" ||
		language(*lang) != languageGo ||
		*packageName == "" ||
		*funcName == "" ||
		*outputFile == "" {
		flag.Usage()
		os.Exit(1)
	}

	err := gen.GenerateFile(*inputJSON, *packageName, *funcName, *outputFile)
	if err != nil {
		log.Fatal(err)
	}
}
