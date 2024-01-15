package gen

import (
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
