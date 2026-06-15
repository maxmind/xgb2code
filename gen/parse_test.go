package gen

import (
	"encoding/json"
	"math"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseTreeInfo(t *testing.T) {
	x, err := readModel(filepath.Join("testdata", "small-model", "model.json"))
	require.NoError(t, err)

	treeInfo, err := parseTreeInfo(
		x.Learner.GradientBooster.Model.Trees[0],
	)
	require.NoError(t, err)

	assert.Equal(
		t,
		&node{
			data: nodeData{
				DefaultLeft:    1,
				ID:             0,
				SplitCondition: 102.5,
				SplitIndex:     22,
			},
			left: &node{
				data: nodeData{
					DefaultLeft:    1,
					ID:             1,
					SplitCondition: 698.8,
					SplitIndex:     3,
				},
				left: &node{
					data: nodeData{
						DefaultLeft:    1,
						ID:             3,
						SplitCondition: 0.1789,
						SplitIndex:     24,
					},
					left: &node{
						data: nodeData{
							DefaultLeft:    1,
							ID:             7,
							SplitCondition: 92.68,
							SplitIndex:     2,
						},
						left: &node{
							data: nodeData{
								DefaultLeft:    0,
								ID:             11,
								SplitCondition: 0.37818182,
								SplitIndex:     0,
							},
						},
						right: &node{
							data: nodeData{
								DefaultLeft:    0,
								ID:             12,
								SplitCondition: 0.1,
								SplitIndex:     0,
							},
						},
					},
					right: &node{
						data: nodeData{
							DefaultLeft:    0,
							ID:             8,
							SplitCondition: 0,
							SplitIndex:     0,
						},
					},
				},
				right: &node{
					data: nodeData{
						DefaultLeft:    0,
						ID:             4,
						SplitCondition: -0.13333334,
						SplitIndex:     0,
					},
				},
			},
			right: &node{
				data: nodeData{
					DefaultLeft:    0,
					ID:             2,
					SplitCondition: 0.08855,
					SplitIndex:     4,
				},
				left: &node{
					data: nodeData{
						DefaultLeft:    0,
						ID:             5,
						SplitCondition: -0.03636364,
						SplitIndex:     0,
					},
				},
				right: &node{
					data: nodeData{
						DefaultLeft:    0,
						ID:             6,
						SplitCondition: 0.6218,
						SplitIndex:     11,
					},
					left: &node{
						data: nodeData{
							DefaultLeft:    0,
							ID:             9,
							SplitCondition: 0,
							SplitIndex:     0,
						},
					},
					right: &node{
						data: nodeData{
							DefaultLeft:    0,
							ID:             10,
							SplitCondition: -0.37037036,
							SplitIndex:     0,
						},
					},
				},
			},
		},
		treeInfo,
	)
}

func TestReadModelMeta(t *testing.T) {
	tests := []struct {
		name              string
		objectiveName     string
		baseScore         string
		numTarget         string
		numClass          string
		expectedObjective objective
		expectedIntercept float64
		expectError       bool
	}{
		{
			name:              "binary:logistic with default base_score",
			objectiveName:     "binary:logistic",
			baseScore:         "5E-1",
			expectedObjective: objectiveBinaryLogistic,
			expectedIntercept: 0.0,
		},
		{
			name:              "reg:logistic with default base_score",
			objectiveName:     "reg:logistic",
			baseScore:         "0.5",
			expectedObjective: objectiveRegLogistic,
			expectedIntercept: 0.0,
		},
		{
			// reg:logistic shares the sigmoid path with binary:logistic, so a
			// non-default base_score must be logit-transformed, not used raw.
			name:              "reg:logistic with non-default base_score",
			objectiveName:     "reg:logistic",
			baseScore:         "0.7",
			expectedObjective: objectiveRegLogistic,
			expectedIntercept: math.Log(0.7 / 0.3),
		},
		{
			name:              "reg:squarederror with default base_score",
			objectiveName:     "reg:squarederror",
			baseScore:         "0.5",
			expectedObjective: objectiveRegSquaredError,
			expectedIntercept: 0.5,
		},
		{
			name:              "reg:squarederror with non-default base_score",
			objectiveName:     "reg:squarederror",
			baseScore:         "150",
			expectedObjective: objectiveRegSquaredError,
			expectedIntercept: 150.0,
		},
		{
			name:              "binary:logistic with non-default base_score",
			objectiveName:     "binary:logistic",
			baseScore:         "0.7",
			expectedObjective: objectiveBinaryLogistic,
			expectedIntercept: math.Log(0.7 / 0.3),
		},
		{
			name:              "reg:squarederror with vector-format base_score",
			objectiveName:     "reg:squarederror",
			baseScore:         "[1.5E2]",
			expectedObjective: objectiveRegSquaredError,
			expectedIntercept: 150.0,
		},
		{
			// Unlike binary:logistic, logitraw outputs the raw margin and uses
			// base_score directly (matching XGBoost), so no logit transform.
			name:              "binary:logitraw uses a raw base_score intercept",
			objectiveName:     "binary:logitraw",
			baseScore:         "0.7",
			expectedObjective: objectiveBinaryLogitRaw,
			expectedIntercept: 0.7,
		},
		{
			// Because base_score is used raw, values outside (0, 1) that would
			// be rejected for binary:logistic are valid for logitraw.
			name:              "binary:logitraw accepts base_score outside (0,1)",
			objectiveName:     "binary:logitraw",
			baseScore:         "1.5",
			expectedObjective: objectiveBinaryLogitRaw,
			expectedIntercept: 1.5,
		},
		{
			name:              "reg:linear uses a raw base_score intercept",
			objectiveName:     "reg:linear",
			baseScore:         "150",
			expectedObjective: objectiveRegLinear,
			expectedIntercept: 150.0,
		},
		{
			name:              "reg:absoluteerror uses a raw base_score intercept",
			objectiveName:     "reg:absoluteerror",
			baseScore:         "2.5",
			expectedObjective: objectiveRegAbsoluteError,
			expectedIntercept: 2.5,
		},
		{
			name:              "reg:pseudohubererror uses a raw base_score intercept",
			objectiveName:     "reg:pseudohubererror",
			baseScore:         "-3",
			expectedObjective: objectiveRegPseudoHuberError,
			expectedIntercept: -3.0,
		},
		{
			name:          "unsupported objective",
			objectiveName: "multi:softprob",
			baseScore:     "0.5",
			expectError:   true,
		},
		{
			name:          "logistic with base_score 0",
			objectiveName: "binary:logistic",
			baseScore:     "0",
			expectError:   true,
		},
		{
			name:          "logistic with base_score 1",
			objectiveName: "binary:logistic",
			baseScore:     "1",
			expectError:   true,
		},
		{
			name:          "logistic with base_score NaN",
			objectiveName: "binary:logistic",
			baseScore:     "NaN",
			expectError:   true,
		},
		{
			name:          "logistic with base_score Inf",
			objectiveName: "binary:logistic",
			baseScore:     "Infinity",
			expectError:   true,
		},
		{
			name:          "reg:squarederror with base_score Inf",
			objectiveName: "reg:squarederror",
			baseScore:     "Infinity",
			expectError:   true,
		},
		{
			name:          "missing base_score",
			objectiveName: "binary:logistic",
			baseScore:     "",
			expectError:   true,
		},
		{
			name:          "base_score with unmatched leading bracket",
			objectiveName: "reg:squarederror",
			baseScore:     "[1.5",
			expectError:   true,
		},
		{
			name:          "base_score with unmatched trailing bracket",
			objectiveName: "reg:squarederror",
			baseScore:     "1.5]",
			expectError:   true,
		},
		{
			name:              "single-quantile reg:quantileerror uses a raw intercept",
			objectiveName:     "reg:quantileerror",
			baseScore:         "[1.405E2]",
			numTarget:         "1",
			expectedObjective: objectiveRegQuantileError,
			expectedIntercept: 140.5,
		},
		{
			name:          "multi-quantile reg:quantileerror is rejected",
			objectiveName: "reg:quantileerror",
			baseScore:     "[8.675E1,1.405E2,2.125E2]",
			numTarget:     "3",
			expectError:   true,
		},
		{
			name:          "multi-target reg:squarederror is rejected",
			objectiveName: "reg:squarederror",
			baseScore:     "[1.5E2]",
			numTarget:     "2",
			expectError:   true,
		},
		{
			name:          "multi-class num_class is rejected",
			objectiveName: "reg:squarederror",
			baseScore:     "0.5",
			numClass:      "3",
			expectError:   true,
		},
		{
			name:          "non-numeric num_target is rejected",
			objectiveName: "reg:squarederror",
			baseScore:     "0.5",
			numTarget:     "not-a-number",
			expectError:   true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			x := &xgbModel{}
			x.Learner.Objective.Name = test.objectiveName
			x.Learner.LearnerModelParam.BaseScore = test.baseScore
			x.Learner.LearnerModelParam.NumTarget = test.numTarget
			x.Learner.LearnerModelParam.NumClass = test.numClass

			meta, err := readModelMeta(x)
			if test.expectError {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, test.expectedObjective, meta.objective)
			assert.InDelta(t, test.expectedIntercept, meta.intercept, 1e-10)
		})
	}
}

func TestReadTreesNumParallelTree(t *testing.T) {
	// makeModel builds a model with a single minimal leaf tree so parseTreeInfo
	// succeeds, then sets the fields the num_parallel_tree guard depends on.
	makeModel := func(bestIteration, numParallelTree string) *xgbModel {
		x := &xgbModel{}
		x.Learner.Attributes.BestIteration = json.Number(bestIteration)
		x.Learner.GradientBooster.Model.GBTreeModelParam.NumParallelTree = numParallelTree
		tree := xgbTree{
			DefaultLeft:     []int{0},
			LeftChildren:    []int{-1},
			RightChildren:   []int{-1},
			SplitConditions: []float64{0},
			SplitIndices:    []int{0},
		}
		tree.TreeParam.NumNodes = "1"
		x.Learner.GradientBooster.Model.Trees = []xgbTree{tree}
		return x
	}

	tests := []struct {
		name            string
		bestIteration   string
		numParallelTree string
		expectError     bool
	}{
		{
			// Without best_iteration every tree is kept, which is correct even
			// for forests, so the guard must not fire.
			name:            "no best_iteration keeps all trees regardless of num_parallel_tree",
			bestIteration:   "",
			numParallelTree: "2",
		},
		{
			name:            "single tree per iteration is allowed",
			bestIteration:   "0",
			numParallelTree: "1",
		},
		{
			name:            "absent num_parallel_tree is allowed",
			bestIteration:   "0",
			numParallelTree: "",
		},
		{
			name:            "num_parallel_tree > 1 with best_iteration is rejected",
			bestIteration:   "0",
			numParallelTree: "2",
			expectError:     true,
		},
		{
			name:            "non-numeric num_parallel_tree is rejected",
			bestIteration:   "0",
			numParallelTree: "not-a-number",
			expectError:     true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			trees, err := readTrees(
				makeModel(test.bestIteration, test.numParallelTree),
			)
			if test.expectError {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			assert.Len(t, trees, 1)
		})
	}
}

func TestReadModelMetaFromJSON(t *testing.T) {
	x, err := readModel(filepath.Join("testdata", "small-model", "model.json"))
	require.NoError(t, err)

	meta, err := readModelMeta(x)
	require.NoError(t, err)

	assert.Equal(t, objectiveBinaryLogistic, meta.objective)
	assert.InDelta(t, 0.0, meta.intercept, 1e-10)
}

func BenchmarkParseTreeInfo(b *testing.B) {
	x, err := readModel(filepath.Join("testdata", "small-model", "model.json"))
	require.NoError(b, err)
	tree := x.Learner.GradientBooster.Model.Trees[0]

	for b.Loop() {
		_, err := parseTreeInfo(tree)
		require.NoError(b, err)
	}
}
