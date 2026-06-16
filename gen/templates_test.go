package gen

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCategoryTest(t *testing.T) {
	tests := []struct {
		name string
		data nodeData
		want string
	}{
		{
			name: "numeric split has no category test",
			data: nodeData{SplitCondition: 1.5},
			want: "",
		},
		{
			// An empty set means no value routes right, so every present value
			// routes left; the predicate must be the constant "true".
			name: "empty category set",
			data: nodeData{Categorical: true, SplitIndex: 3},
			want: "true",
		},
		{
			name: "single category",
			data: nodeData{Categorical: true, SplitIndex: 1, Categories: []int{2}},
			want: "(*data[1] != 2)",
		},
		{
			name: "multiple categories",
			data: nodeData{Categorical: true, SplitIndex: 3, Categories: []int{0, 2, 5}},
			want: "(*data[3] != 0 && *data[3] != 2 && *data[3] != 5)",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			assert.Equal(t, test.want, categoryTest(test.data))
		})
	}
}
