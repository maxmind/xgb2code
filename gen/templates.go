package gen

import (
	"bytes"
	_ "embed"
	"fmt"
	"strings"
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

// categoryTest builds the "value not in category set" predicate for a
// categorical split, e.g. "(*data[3] != 0 && *data[3] != 2)". It returns the
// empty string for numeric splits. A categorical node with an empty set routes
// every present value left, so the predicate is the constant "true".
func categoryTest(d nodeData) string {
	if !d.Categorical {
		return ""
	}
	if len(d.Categories) == 0 {
		return "true"
	}
	parts := make([]string, len(d.Categories))
	for i, c := range d.Categories {
		parts[i] = fmt.Sprintf("*data[%d] != %d", d.SplitIndex, c)
	}
	return "(" + strings.Join(parts, " && ") + ")"
}

func indent(level int) string {
	var sb strings.Builder
	for range level + 1 {
		sb.WriteString("\t")
	}
	return sb.String()
}

type decisionNodeParams struct {
	nodeData

	Left  string
	Right string
	Level int
	// CategoryTest is the Go expression, set only for categorical splits, that
	// is true when the feature value is *not* in the node's category set (and so
	// routes left). It mirrors the numeric "*data[i] < threshold" predicate.
	CategoryTest string
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
			Left:         left,
			Level:        level,
			nodeData:     tree.data,
			Right:        right,
			CategoryTest: categoryTest(tree.data),
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
	Intercept     float64
	PackageName   string
	TreeFunctions []treeFunction
	UseSigmoid    bool
}

func (r *renderer) executeRoot(
	packageName,
	funcName string,
	treeFunctions []treeFunction,
	meta modelMeta,
) (string, error) {
	var buf bytes.Buffer
	err := r.template.ExecuteTemplate(
		&buf,
		rootTemplateName,
		rootParams{
			FuncName:      funcName,
			Intercept:     meta.intercept,
			PackageName:   packageName,
			TreeFunctions: treeFunctions,
			UseSigmoid:    meta.useSigmoid,
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
