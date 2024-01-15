package gen

import (
	"bytes"
	_ "embed"
	"fmt"
	"text/template"
)

type renderer struct {
	template *template.Template
}

//go:embed templates/decision_node.tmpl
var decisionNodeTemplate string

//go:embed templates/root.tmpl
var rootTemplate string

//go:embed templates/terminal_node.tmpl
var terminalNodeTemplate string

const (
	decisionNodeTemplateName = "decision_node"
	rootTemplateName         = "root"
	terminalNodeTemplateName = "terminal_node"
)

var templateNameToTemplate = map[string]string{
	decisionNodeTemplateName: decisionNodeTemplate,
	rootTemplateName:         rootTemplate,
	terminalNodeTemplateName: terminalNodeTemplate,
}

func newRenderer() (*renderer, error) {
	var r renderer

	fm := template.FuncMap{
		"indent": indent,
	}

	r.template = template.New(rootTemplateName).Funcs(fm)

	for name, body := range templateNameToTemplate {
		if _, err := r.template.New(name).Parse(body); err != nil {
			return nil, fmt.Errorf(
				"error parsing %s template: %w",
				name,
				err,
			)
		}
	}
	return &r, nil
}

func indent(level int) string {
	var s string
	for i := 0; i < level+1; i++ {
		s += "\t"
	}
	return s
}

type decisionNodeParams struct {
	nodeData
	Left  string
	Level int
	Right string
}

func (r *renderer) executeDecisionNode(
	tree *node,
	level int,
	left,
	right string,
) (string, error) {
	var buf bytes.Buffer
	err := r.template.ExecuteTemplate(
		&buf,
		decisionNodeTemplateName,
		decisionNodeParams{
			Left:     left,
			Level:    level,
			nodeData: tree.data,
			Right:    right,
		},
	)
	if err != nil {
		return "", fmt.Errorf(
			"error executing decision_node template: %w",
			err,
		)
	}
	return buf.String(), nil
}

type rootParams struct {
	FuncName      string
	PackageName   string
	TreeFunctions []treeFunction
}

func (r *renderer) executeRoot(
	packageName,
	funcName string,
	treeFunctions []treeFunction,
) (string, error) {
	var buf bytes.Buffer
	err := r.template.ExecuteTemplate(
		&buf,
		rootTemplateName,
		rootParams{
			FuncName:      funcName,
			PackageName:   packageName,
			TreeFunctions: treeFunctions,
		},
	)
	if err != nil {
		return "", fmt.Errorf(
			"error executing root template: %w",
			err,
		)
	}
	return buf.String(), nil
}

type terminalNodeParams struct {
	nodeData
	Level int
}

func (r *renderer) executeTerminalNode(
	tree *node,
	level int,
) (string, error) {
	var buf bytes.Buffer
	err := r.template.ExecuteTemplate(
		&buf,
		terminalNodeTemplateName,
		terminalNodeParams{
			Level:    level,
			nodeData: tree.data,
		},
	)
	if err != nil {
		return "", fmt.Errorf(
			"error executing terminal_node template: %w",
			err,
		)
	}
	return buf.String(), nil
}
