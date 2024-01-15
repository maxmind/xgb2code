package main

import (
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/maxmind/xgb2code/gen"
	"github.com/stretchr/testify/require"
)

func TestGenerateAndRunModels(t *testing.T) {
	tests := []struct {
		model string
	}{
		{model: "small-model"},
		{model: "breast-cancer"},
	}

	for _, test := range tests {
		t.Run(test.model, func(t *testing.T) {
			// Generate the code.
			modelDir := filepath.Join("gen", "testdata", test.model)
			modelFile := filepath.Join(modelDir, "model.json")

			packageName := "main"
			functionName := "predict"

			outputDir := t.TempDir()
			funcFile := filepath.Join(outputDir, "predict.go")

			err := gen.GenerateFile(modelFile, packageName, functionName, funcFile)
			require.NoError(t, err)

			// Copy the test program and test data into place.

			files := []string{
				filepath.Join("gen", "testdata", "main.go"),
				filepath.Join(modelDir, "xtest.csv"),
				filepath.Join(modelDir, "preds.csv"),
			}

			for _, file := range files {
				ifh, err := os.Open(filepath.Clean(file))
				require.NoError(t, err)

				tempFile := filepath.Join(outputDir, filepath.Base(file))

				ofh, err := os.Create(filepath.Clean(tempFile))
				require.NoError(t, err)

				_, err = io.Copy(ofh, ifh)
				require.NoError(t, err)

				err = ifh.Close()
				require.NoError(t, err)

				err = ofh.Close()
				require.NoError(t, err)
			}

			// Run the program.

			cmd := exec.Command("go", "run", "main.go", "predict.go")
			cmd.Dir = outputDir

			buf, err := cmd.CombinedOutput()
			require.NoError(t, err, string(buf))
		})
	}
}
