package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
)

// Run the model in predict() using features from the input CSV for the model
// (xtest.csv) and check the predictions match what we expect in the output CSV
// (preds.csv).
func main() {
	if err := testModel(); err != nil {
		log.Fatal(err)
	}
}

func testModel() error {
	featuresRows, err := readCSV("xtest.csv")
	if err != nil {
		return err
	}

	predictionsRows, err := readCSV("preds.csv")
	if err != nil {
		return err
	}

	for i := range featuresRows {
		if err := testPrediction(featuresRows[i], predictionsRows[i][0]); err != nil {
			return err
		}
	}

	return nil
}

func readCSV(path string) ([][]string, error) {
	fh, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer fh.Close()

	rows, err := csv.NewReader(fh).ReadAll()
	if err != nil {
		return nil, err
	}

	return rows, nil
}

func testPrediction(
	featuresRow []string,
	predictionStr string,
) error {
	// Parse features into *float32s.

	var features []*float32
	for _, featureStr := range featuresRow {
		if featureStr == "" {
			features = append(features, nil)
			continue
		}

		feature64, err := strconv.ParseFloat(featureStr, 64)
		if err != nil {
			return err
		}
		feature := float32(feature64)
		features = append(features, &feature)
	}

	// Parse prediction into a float32.

	prediction64, err := strconv.ParseFloat(predictionStr, 64)
	if err != nil {
		return err
	}
	expectedPrediction := float32(prediction64)

	// Run the model.

	gotPrediction := predict(features, false)

	// Allow for float32 rounding differences between XGBoost's prediction and
	// the generated code. Regression objectives can produce large-magnitude
	// outputs where a tight absolute bound is unrealistic, so accept the
	// prediction if either the absolute or the relative error is small. The
	// relative denominator is floored at 1.0 so that for small-magnitude
	// outputs (the [0, 1] range of logistic objectives, or near-zero
	// regression targets) the check stays effectively absolute, rather than a
	// target near zero blowing the relative error up to infinity.
	const tolerance = 0.00001
	absDelta := math.Abs(float64(gotPrediction - expectedPrediction))
	relDelta := absDelta / math.Max(math.Abs(float64(expectedPrediction)), 1.0)
	if absDelta > tolerance && relDelta > tolerance {
		return fmt.Errorf("got %f, expected %f", gotPrediction, expectedPrediction)
	}

	return nil
}
