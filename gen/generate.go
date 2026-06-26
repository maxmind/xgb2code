// Package gen generates Go code from an XGBoost model.
package gen

import (
	"fmt"
	"go/format"
	"math"
	"os"
)

type treeFunction struct {
	Code string
	Name string
}

func generateSource(
	packageName,
	funcName string,
	trees []*node,
	meta modelMeta,
	r *renderer,
) (string, error) {
	var treeFunctions []treeFunction
	for i, t := range trees {
		level := 0
		code, err := codegenTree(r, t, level)
		if err != nil {
			return "", err
		}

		treeFunctions = append(
			treeFunctions,
			treeFunction{
				Code: code,
				Name: fmt.Sprintf("tree%s_%d", funcName, i),
			},
		)
	}

	code, err := r.executeRoot(packageName, funcName, treeFunctions, meta)
	if err != nil {
		return "", err
	}

	// We run the code through formatting to catch structural/syntax errors. We
	// don't return the formatted code since we intend what we generate to already
	// be well formatted. Note this only validates syntax, not values: a stray
	// non-finite literal like -Inf is a valid Go identifier and would pass here,
	// so checkSplitCondition (not this) is what guarantees no such literal is
	// ever emitted.
	if _, err := format.Source([]byte(code)); err != nil {
		return "", fmt.Errorf("error formatting code: %w", err)
	}

	return code, nil
}

func codegenTree(r *renderer, tree *node, level int) (string, error) {
	isLeaf := tree.left == nil || tree.right == nil
	if isLeaf {
		return r.executeTerminalNode(tree, level)
	}

	// A -Infinity threshold on a numeric split makes "*data[i] < threshold"
	// false for every present value, so the split routes purely on whether the
	// feature is missing. Rendering it literally would emit the uncompilable
	// "*data[i] < -Inf", so collapse it to the equivalent missingness branch.
	if !tree.data.Categorical && math.IsInf(tree.data.SplitCondition, -1) {
		return codegenMissingnessSplit(r, tree, level)
	}

	left, err := codegenTree(r, tree.left, level+1)
	if err != nil {
		return "", err
	}
	right, err := codegenTree(r, tree.right, level+1)
	if err != nil {
		return "", err
	}

	return r.executeDecisionNode(tree, level, left, right, false)
}

// codegenMissingnessSplit emits code for a numeric node whose -Infinity threshold
// makes it route on missingness alone: missing values go left (if default_left)
// and every present value goes right.
func codegenMissingnessSplit(r *renderer, tree *node, level int) (string, error) {
	// With default_left == 0, missing also routes right, so the node reduces to
	// its right subtree. The left subtree is unreachable and dropped; that is safe
	// because parseTreeInfo has already validated every node before codegen runs.
	if tree.data.DefaultLeft == 0 {
		return codegenTree(r, tree.right, level)
	}

	left, err := codegenTree(r, tree.left, level+1)
	if err != nil {
		return "", err
	}
	right, err := codegenTree(r, tree.right, level+1)
	if err != nil {
		return "", err
	}

	return r.executeDecisionNode(tree, level, left, right, true)
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

	meta, err := readModelMeta(x)
	if err != nil {
		return err
	}

	r, err := newRenderer()
	if err != nil {
		return err
	}

	code, err := generateSource(packageName, funcName, trees, meta, r)
	if err != nil {
		return err
	}

	if err := os.WriteFile(outputFile, []byte(code), 0o644); err != nil {
		return fmt.Errorf("error writing file: %w", err)
	}
	return nil
}
