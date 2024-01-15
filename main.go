// xgb2code generates code for an XGB model.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
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

	err := GenerateFile(*inputJSON, *packageName, *funcName, *outputFile)
	if err != nil {
		log.Fatal(err)
	}
}

// GenerateFile generates a .go file containing a function that implements the XGB model.
func GenerateFile(
	inputJSON string,
	packageName,
	funcName,
	outputFile string,
) error {
	x, err := readModel(inputJSON)
	if err != nil {
		return err
	}

	trees, err := readTrees(x)
	if err != nil {
		return err
	}

	r, err := newRenderer()
	if err != nil {
		return err
	}

	code, err := generateSource(packageName, funcName, trees, r)
	if err != nil {
		return err
	}

	if err := os.WriteFile(outputFile, []byte(code), 0o644); err != nil {
		return fmt.Errorf("error writing file: %w", err)
	}

	return nil
}
