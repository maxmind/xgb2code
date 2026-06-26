package gen

import (
	"math"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCodegenMissingnessSplit(t *testing.T) {
	// leaf builds a terminal node with the given output value.
	leaf := func(v float64) *node {
		return &node{data: nodeData{SplitCondition: v}}
	}

	// negInfRoot is a decision node whose -Infinity threshold makes it route on
	// missingness alone. defaultLeft controls where missing values go.
	negInfRoot := func(defaultLeft int) *node {
		return &node{
			data: nodeData{
				SplitCondition: math.Inf(-1),
				SplitIndex:     0,
				DefaultLeft:    defaultLeft,
			},
			left:  leaf(10),
			right: leaf(20),
		}
	}

	t.Run("default_left routes missing left, present right", func(t *testing.T) {
		r, err := newRenderer()
		require.NoError(t, err)

		code, err := codegenTree(r, negInfRoot(1), 0)
		require.NoError(t, err)

		// Verify branch placement, not just presence: missing (nil) routes to
		// the left leaf (10) and every present value to the right leaf (20). An
		// inverted condition or swapped leaves would slip past a presence-only
		// check but not these ordering assertions.
		nilIdx := strings.Index(code, "if data[0] == nil {")
		elseIdx := strings.Index(code, "} else {")
		leftIdx := strings.Index(code, "sum += 10")
		rightIdx := strings.Index(code, "sum += 20")
		require.GreaterOrEqual(t, nilIdx, 0, "missingness branch must test data[0] == nil")
		require.Greater(t, elseIdx, nilIdx)
		assert.True(t, leftIdx > nilIdx && leftIdx < elseIdx, "missing must add the left leaf (10)")
		assert.Greater(t, rightIdx, elseIdx, "present must add the right leaf (20)")
		// The uncompilable literal comparison must never be emitted.
		assert.NotContains(t, code, "Inf")
	})

	t.Run("no default_left collapses to the right subtree", func(t *testing.T) {
		r, err := newRenderer()
		require.NoError(t, err)

		code, err := codegenTree(r, negInfRoot(0), 0)
		require.NoError(t, err)

		// Present and missing both route right, so the node disappears entirely.
		assert.Equal(t, "sum += 20", strings.TrimSpace(code))
	})

	t.Run("no default_left collapses into a right decision subtree", func(t *testing.T) {
		r, err := newRenderer()
		require.NoError(t, err)

		// The surviving right child is itself a decision node. The collapse must
		// render it at the dropped node's level (not level+1) and omit the
		// unreachable left subtree. A leaf-only right child (the case above)
		// cannot catch a re-indentation bug because it has no indentation.
		root := &node{
			data: nodeData{SplitCondition: math.Inf(-1), DefaultLeft: 0, SplitIndex: 0},
			left: leaf(10),
			right: &node{
				data:  nodeData{SplitCondition: 1.5, SplitIndex: 1, DefaultLeft: 1},
				left:  leaf(20),
				right: leaf(30),
			},
		}
		code, err := codegenTree(r, root, 0)
		require.NoError(t, err)

		assert.NotContains(t, code, "sum += 10", "dropped left subtree must not appear")
		// Rendered at the parent's level (0) => a single leading tab, not two.
		assert.True(
			t,
			strings.HasPrefix(code, "\tif data[1] == nil || *data[1] < 1.5 {"),
			"right subtree must render at the parent level; got:\n%s", code,
		)
	})
}
