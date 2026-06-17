package gen

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

type objective string

const (
	objectiveBinaryLogistic      objective = "binary:logistic"
	objectiveBinaryLogitRaw      objective = "binary:logitraw"
	objectiveRegLogistic         objective = "reg:logistic"
	objectiveRegSquaredError     objective = "reg:squarederror"
	objectiveRegLinear           objective = "reg:linear"
	objectiveRegAbsoluteError    objective = "reg:absoluteerror"
	objectiveRegPseudoHuberError objective = "reg:pseudohubererror"
	objectiveRegQuantileError    objective = "reg:quantileerror"
)

type modelMeta struct {
	objective objective
	// intercept is the base score expressed in raw margin (pre-activation)
	// space: logit(base_score) for sigmoid objectives, the raw base_score
	// otherwise. It is added to the summed tree outputs before any sigmoid.
	intercept float64
	// useSigmoid reports whether the summed margin (plus intercept) is passed
	// through a sigmoid to produce the final prediction. It is derived from the
	// objective in readModelMeta, the single place that classifies objectives.
	useSigmoid bool
}

type xgbModel struct {
	Learner struct {
		Attributes struct {
			BestIteration json.Number `json:"best_iteration"`
		} `json:"attributes"`
		GradientBooster struct {
			Model struct {
				GBTreeModelParam struct {
					NumParallelTree string `json:"num_parallel_tree"`
				} `json:"gbtree_model_param"`
				Trees []xgbTree `json:"trees"`
			} `json:"model"`
		} `json:"gradient_booster"`
		LearnerModelParam struct {
			BaseScore string `json:"base_score"`
			NumClass  string `json:"num_class"`
			NumTarget string `json:"num_target"`
		} `json:"learner_model_param"`
		Objective struct {
			Name string `json:"name"`
		} `json:"objective"`
	} `json:"learner"`
}

type xgbTree struct {
	DefaultLeft     []int     `json:"default_left"`
	LeftChildren    []int     `json:"left_children"`
	RightChildren   []int     `json:"right_children"`
	SplitConditions []float64 `json:"split_conditions"`
	SplitIndices    []int     `json:"split_indices"`
	// SplitType marks each node's split kind: 0 = numeric, 1 = categorical. It
	// is absent in models trained without categorical features, in which case
	// every split is numeric.
	SplitType []int `json:"split_type"`
	// The fields below describe categorical splits. Categories is the flattened
	// list of category values across all categorical nodes; for each entry k in
	// CategoriesNodes (a node ID), the values that route to that node's right
	// child are Categories[CategoriesSegments[k] : CategoriesSegments[k]+CategoriesSizes[k]].
	Categories         []int `json:"categories"`
	CategoriesNodes    []int `json:"categories_nodes"`
	CategoriesSegments []int `json:"categories_segments"`
	CategoriesSizes    []int `json:"categories_sizes"`
	TreeParam          struct {
		NumNodes json.Number `json:"num_nodes"`
	} `json:"tree_param"`
}

func readModel(inputJSON string) (*xgbModel, error) {
	fh, err := os.Open(filepath.Clean(inputJSON))
	if err != nil {
		return nil, fmt.Errorf("error opening file: %w", err)
	}
	defer fh.Close()

	var x xgbModel
	if err := json.NewDecoder(fh).Decode(&x); err != nil {
		return nil, fmt.Errorf("error decoding JSON: %w", err)
	}

	return &x, nil
}

func readTrees(x *xgbModel) ([]*node, error) {
	var trees []*node
	for i := range x.Learner.GradientBooster.Model.Trees {
		t := x.Learner.GradientBooster.Model.Trees[i]

		treeInfo, err := parseTreeInfo(t)
		if err != nil {
			return nil, err
		}

		trees = append(trees, treeInfo)
	}

	// Models trained without early stopping have no best_iteration; keep all
	// trees. Summing every tree is what XGBoost does at inference, so this is
	// correct regardless of num_parallel_tree.
	if x.Learner.Attributes.BestIteration == "" {
		return trees, nil
	}

	bestIteration, err := x.Learner.Attributes.BestIteration.Int64()
	if err != nil {
		return nil, fmt.Errorf(
			"error parsing best_iteration as int: %w",
			err,
		)
	}

	// Truncating to best_iteration assumes one tree per boosting round, so the
	// (bestIteration+1)th tree is the last tree to keep. Models with
	// num_parallel_tree > 1 (e.g. boosted random forests) emit several trees
	// per round, so this slice would keep the wrong subset; reject them rather
	// than silently produce wrong predictions.
	if err := checkSingleTreePerIteration(x); err != nil {
		return nil, err
	}

	return trees[:bestIteration+1], nil
}

// checkSingleTreePerIteration rejects models that emit more than one tree per
// boosting round (num_parallel_tree > 1), e.g. boosted random forests. The
// best_iteration truncation in readTrees assumes one tree per round, so such
// models cannot be truncated correctly. An empty value means the field is
// absent (one tree per round).
func checkSingleTreePerIteration(x *xgbModel) error {
	v := x.Learner.GradientBooster.Model.GBTreeModelParam.NumParallelTree
	if v == "" {
		return nil
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return fmt.Errorf("error parsing num_parallel_tree: %w", err)
	}
	if n > 1 {
		return fmt.Errorf(
			"models with num_parallel_tree > 1 are not supported "+
				"(num_parallel_tree = %d)",
			n,
		)
	}
	return nil
}

type node struct {
	left  *node
	right *node
	data  nodeData
}

type nodeData struct {
	DefaultLeft    int
	ID             int64
	SplitCondition float64
	SplitIndex     int
	// Categorical reports whether this is a categorical split. When true,
	// Categories holds the category values that route to the right child and
	// SplitCondition is unused (XGBoost stores a dummy threshold there).
	Categorical bool
	Categories  []int
}

func parseTreeInfo(xt xgbTree) (*node, error) {
	numNodes, err := xt.TreeParam.NumNodes.Int64()
	if err != nil {
		return nil, fmt.Errorf(
			"error parsing num_nodes as an integer: %w",
			err,
		)
	}

	if err := checkNodeArrays(xt, numNodes); err != nil {
		return nil, err
	}

	categories, err := categorySets(xt, numNodes)
	if err != nil {
		return nil, err
	}

	nodes := make([]node, numNodes)
	for i := range numNodes {
		cats, categorical := categories[i]
		nodes[i].data = nodeData{
			DefaultLeft:    xt.DefaultLeft[i],
			ID:             i,
			SplitCondition: xt.SplitConditions[i],
			SplitIndex:     xt.SplitIndices[i],
			Categorical:    categorical,
			Categories:     cats,
		}

		left := xt.LeftChildren[i]
		right := xt.RightChildren[i]

		// A leaf has no children (both -1). Any other combination, including a
		// half-wired node with exactly one child, is malformed: the codegen
		// treats a node with a nil child as a leaf and would silently drop the
		// other subtree, so reject it rather than mispredict.
		if left == -1 && right == -1 {
			continue
		}
		if left < 0 || int64(left) >= numNodes ||
			right < 0 || int64(right) >= numNodes {
			return nil, fmt.Errorf(
				"node %d has out-of-range children "+
					"(left=%d, right=%d, num_nodes=%d)",
				i,
				left,
				right,
				numNodes,
			)
		}

		nodes[i].left = &nodes[left]
		nodes[i].right = &nodes[right]
	}

	return &nodes[0], nil // Root node
}

// checkNodeArrays verifies that every per-node array has exactly one entry per
// node, so the indexing in parseTreeInfo cannot panic on a truncated or
// inconsistent model. num_nodes comes from a separate JSON field and so is not
// inherently consistent with the arrays it describes.
func checkNodeArrays(xt xgbTree, numNodes int64) error {
	arrays := []struct {
		name string
		n    int
	}{
		{"default_left", len(xt.DefaultLeft)},
		{"left_children", len(xt.LeftChildren)},
		{"right_children", len(xt.RightChildren)},
		{"split_conditions", len(xt.SplitConditions)},
		{"split_indices", len(xt.SplitIndices)},
	}
	for _, a := range arrays {
		if int64(a.n) != numNodes {
			return fmt.Errorf(
				"%s has %d entries but num_nodes is %d",
				a.name,
				a.n,
				numNodes,
			)
		}
	}
	return nil
}

// categorySets maps each categorical node's ID to the category values that
// route to its right child, decoding XGBoost's flattened categories/segments/
// sizes representation. It returns an empty map for models trained without
// categorical features. It validates the arrays rather than trusting them: a
// malformed or inconsistent encoding would otherwise cause a categorical node
// to be silently emitted as a numeric split on its dummy threshold, producing
// wrong predictions.
func categorySets(xt xgbTree, numNodes int64) (map[int64][]int, error) {
	n := len(xt.CategoriesNodes)
	if len(xt.CategoriesSegments) != n || len(xt.CategoriesSizes) != n {
		return nil, fmt.Errorf(
			"inconsistent categorical arrays: categories_nodes=%d, "+
				"categories_segments=%d, categories_sizes=%d",
			n,
			len(xt.CategoriesSegments),
			len(xt.CategoriesSizes),
		)
	}

	sets := make(map[int64][]int, n)
	for k := range n {
		start := xt.CategoriesSegments[k]
		size := xt.CategoriesSizes[k]
		if start < 0 || size < 0 || start > len(xt.Categories)-size {
			return nil, fmt.Errorf(
				"categorical segment [%d:%d+%d] out of range for "+
					"categories of length %d",
				start,
				start,
				size,
				len(xt.Categories),
			)
		}
		nodeID := int64(xt.CategoriesNodes[k])
		if nodeID < 0 || nodeID >= numNodes {
			return nil, fmt.Errorf(
				"categories_nodes[%d] = %d out of range for num_nodes %d",
				k,
				nodeID,
				numNodes,
			)
		}
		cats := make([]int, size)
		copy(cats, xt.Categories[start:start+size])
		sets[nodeID] = cats
	}

	// split_type is the only independent signal of which nodes are categorical,
	// so it is what lets us verify that every categorical node was decoded.
	// Without it we cannot make that check, and a categorical node missing from
	// categories_nodes would be silently emitted as a numeric split on its dummy
	// threshold. Real XGBoost models always include split_type when they have
	// categorical data, so reject categorical data that lacks it rather than
	// risk a wrong prediction.
	if len(xt.SplitType) == 0 {
		if len(sets) > 0 {
			return nil, errors.New(
				"model has categorical data (categories_nodes) but no split_type",
			)
		}
		return sets, nil
	}

	if int64(len(xt.SplitType)) != numNodes {
		return nil, fmt.Errorf(
			"split_type length %d does not match num_nodes %d",
			len(xt.SplitType),
			numNodes,
		)
	}

	// Every node that split_type marks as categorical must have a decoded set,
	// and vice versa; a mismatch means we would treat a node as the wrong split
	// kind. Any split_type other than 0 (numeric) or 1 (categorical) is an
	// encoding we do not understand, so reject it rather than defaulting it to
	// numeric.
	for i := range numNodes {
		switch xt.SplitType[i] {
		case 0, 1:
		default:
			return nil, fmt.Errorf(
				"node %d has unsupported split_type %d",
				i,
				xt.SplitType[i],
			)
		}
		_, hasSet := sets[i]
		isCategorical := xt.SplitType[i] == 1
		if hasSet != isCategorical {
			return nil, fmt.Errorf(
				"node %d has split_type %d but %s in categories_nodes",
				i,
				xt.SplitType[i],
				presence(hasSet),
			)
		}
	}

	return sets, nil
}

func presence(present bool) string {
	if present {
		return "is present"
	}
	return "is absent"
}

func readModelMeta(x *xgbModel) (modelMeta, error) {
	obj := objective(x.Learner.Objective.Name)

	// This switch is the single place that classifies objectives: it both
	// rejects unsupported objectives and decides whether the margin is passed
	// through a sigmoid. Keeping validation and the sigmoid decision together
	// means a newly supported objective cannot be silently treated as
	// non-sigmoid. For sigmoid objectives base_score is a probability that gets
	// converted to a margin-space (logit) intercept; for the rest base_score is
	// used directly as the intercept. Note binary:logitraw, despite being a
	// logistic objective, outputs the raw margin and (as XGBoost does) uses
	// base_score directly, so it is not a sigmoid objective.
	var useSigmoid bool
	switch obj {
	case objectiveBinaryLogistic, objectiveRegLogistic:
		useSigmoid = true
	case objectiveBinaryLogitRaw,
		objectiveRegSquaredError,
		objectiveRegLinear,
		objectiveRegAbsoluteError,
		objectiveRegPseudoHuberError,
		objectiveRegQuantileError:
		useSigmoid = false
	default:
		return modelMeta{}, fmt.Errorf(
			"unsupported objective: %q",
			x.Learner.Objective.Name,
		)
	}

	if err := checkSingleOutput(x); err != nil {
		return modelMeta{}, err
	}

	baseScore, err := parseBaseScore(x.Learner.LearnerModelParam.BaseScore)
	if err != nil {
		return modelMeta{}, fmt.Errorf(
			"error parsing base_score: %w",
			err,
		)
	}

	// For sigmoid objectives base_score is a probability in (0, 1); convert it
	// to the margin-space (logit) intercept the trees operate in. For other
	// objectives base_score is already in prediction space and used directly.
	var intercept float64
	if useSigmoid {
		if baseScore <= 0 || baseScore >= 1 {
			return modelMeta{}, fmt.Errorf(
				"base_score must be between 0 and 1 (exclusive) for %s, got %v",
				obj,
				baseScore,
			)
		}
		intercept = math.Log(baseScore / (1 - baseScore))
	} else {
		intercept = baseScore
	}

	return modelMeta{
		objective:  obj,
		intercept:  intercept,
		useSigmoid: useSigmoid,
	}, nil
}

// checkSingleOutput rejects multi-output models, e.g. multi-target regression
// or multi-quantile reg:quantileerror. The generated code sums all trees into a
// single scalar, so it cannot represent the per-output tree groups that XGBoost
// uses for such models (which would otherwise be silently summed together).
// num_target counts regression/quantile outputs; num_class counts classes.
func checkSingleOutput(x *xgbModel) error {
	p := x.Learner.LearnerModelParam
	if err := checkOutputCount("num_target", p.NumTarget); err != nil {
		return err
	}
	return checkOutputCount("num_class", p.NumClass)
}

// checkOutputCount rejects an output-count field (num_target, num_class) that is
// present and greater than one. An empty value means the field is absent.
func checkOutputCount(name, value string) error {
	if value == "" {
		return nil
	}
	n, err := strconv.Atoi(value)
	if err != nil {
		return fmt.Errorf("error parsing %s: %w", name, err)
	}
	if n > 1 {
		return fmt.Errorf(
			"multi-output models are not supported (%s = %d)",
			name,
			n,
		)
	}
	return nil
}

// parseBaseScore parses the base_score string from the model JSON. Older
// XGBoost versions store this as a plain number (e.g., "5E-1"), while newer
// versions use a single-element vector format (e.g., "[1.5E2]").
func parseBaseScore(s string) (float64, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, errors.New(
			"base_score is missing from the model; the model JSON may be " +
				"from an unsupported XGBoost version",
		)
	}
	// Strip the vector brackets only as a matched pair, so a malformed value
	// with just one bracket (e.g. "[1.5") is rejected rather than silently
	// parsed as the number it surrounds.
	if strings.HasPrefix(s, "[") || strings.HasSuffix(s, "]") {
		if !strings.HasPrefix(s, "[") || !strings.HasSuffix(s, "]") {
			return 0, fmt.Errorf("malformed base_score vector: %q", s)
		}
		s = strings.TrimSpace(s[1 : len(s)-1])
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, fmt.Errorf("not a valid number: %w", err)
	}
	// ParseFloat accepts "NaN", "Inf", etc. These are never valid base
	// scores and would silently produce NaN/Inf predictions (or uncompilable
	// Go code for regression objectives), so reject them explicitly.
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return 0, fmt.Errorf("base_score must be finite, got %v", v)
	}
	return v, nil
}
