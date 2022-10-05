package main

import (
	"fmt"
	"go/format"
)

type treeFunction struct {
	Code string
	Name string
}

func codegen(
	packageName,
	funcName string,
	trees []*node,
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

	code, err := r.executeRoot(packageName, funcName, treeFunctions)
	if err != nil {
		return "", err
	}

	// We run the code through formatting to check for syntax errors. We don't
	// return the formatted code since we intend what we generate to already be
	// well formatted.
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

	left, err := codegenTree(r, tree.left, level+1)
	if err != nil {
		return "", err
	}
	right, err := codegenTree(r, tree.right, level+1)
	if err != nil {
		return "", err
	}

	return r.executeDecisionNode(tree, level, left, right)
}
