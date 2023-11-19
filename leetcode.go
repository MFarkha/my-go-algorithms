package main

import (
	"fmt"
	"math"
)

type ListNode struct {
	Val  int
	Next *ListNode
}

func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	var digits []int
	var digit1, digit2 int
	carry := 0
	for l1 != nil || l2 != nil {
		if l1 != nil {
			digit1 = l1.Val
			l1 = l1.Next
		} else {
			digit1 = 0
		}
		if l2 != nil {
			digit2 = l2.Val
			l2 = l2.Next
		} else {
			digit2 = 0
		}
		if (digit1 + digit2 + carry) >= 10 {
			digits = append(digits, (digit1+digit2+carry)%10)
			carry = 1
		} else {
			digits = append(digits, digit1+digit2+carry)
			carry = 0
		}
		fmt.Printf("digits=%d\n", digits)
	}
	rootNode := &ListNode{Val: digits[0]}
	currentNode := rootNode
	for _, d := range digits[1:] {
		currentNode.Next = &ListNode{Val: d}
		currentNode = currentNode.Next
	}
	return rootNode
}

func romanToInt(s string) int {
	roman_nums := map[string]int{
		"I": 1,
		"V": 5,
		"X": 10,
		"L": 50,
		"C": 100,
		"D": 500,
		"M": 1000,
	}
	prev := string(s[0])
	number := roman_nums[prev]
	for _, r := range s[1:] {
		letter := string(r)
		number += roman_nums[letter]
		if prev == "I" && (letter == "V" || letter == "X") {
			number -= 2
		}
		if prev == "X" && (letter == "L" || letter == "C") {
			number -= 20
		}
		if prev == "C" && (letter == "D" || letter == "M") {
			number -= 200
		}
		prev = letter
	}
	return number
}

func longestCommonPrefix(strs []string) string {
	commonPrefix := func(s1 string, s2 string) (result string) {
		till := min(len(s1), len(s2))
		for i := 0; i < till; i++ {
			if s1[i] != s2[i] {
				break
			}
			result += string(s1[i])
		}
		return
	}
	if len(strs) == 1 {
		return strs[0]
	}
	result := commonPrefix(strs[0], strs[1])
	for _, s := range strs[1:] {
		result = commonPrefix(result, s)
	}
	return result
}

func isValid(s string) bool {
	brackets := map[rune]rune{
		'(': ')',
		'{': '}',
		'[': ']',
	}
	stack := []rune{}
	for _, r := range s {
		if _, ok := brackets[r]; ok {
			stack = append(stack, r)
		} else {
			if len(stack) == 0 || r != brackets[stack[len(stack)-1]] {
				return false
			}
			stack = stack[:len(stack)-1]
		}
	}
	return len(stack) == 0
}

func addBinary(a string, b string) string {
	binary_sum := func(expr string) (string, string) {
		switch {
		case expr == "000":
			return "0", "0"
		case expr == "001" || expr == "010" || expr == "100":
			return "0", "1"
		case expr == "011" || expr == "101" || expr == "110":
			return "1", "0"
		default:
			return "1", "1"
		}
	}
	if len(a) > len(b) {
		till := len(a) - len(b)
		for i := 0; i < till; i++ {
			b = "0" + b
		}
	} else if len(a) < len(b) {
		till := len(b) - len(a)
		for i := 0; i < till; i++ {
			a = "0" + a
		}
	}
	carry := "0"
	var digit, c string
	for i := len(a) - 1; i > -1; i-- {
		carry, digit = binary_sum(carry + string(a[i]) + string(b[i]))
		c = digit + c
	}
	// fmt.Printf("c= %s\n", c)
	if carry == "1" {
		c = "1" + c
	}
	return c
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func (tree *TreeNode) insert(values []int) *TreeNode {
	queue := []*TreeNode{tree}
	i := 0
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		if current.Left == nil {
			if values[i] != 0 {
				// fmt.Printf("i= %d, values = %d\n", i, values[i])
				current.Left = &TreeNode{Val: values[i]}
			}
			i++
			if i >= len(values) {
				return tree
			}
		}
		if current.Left != nil {
			queue = append(queue, current.Left)
		}
		if current.Right == nil {
			if values[i] != 0 {
				// fmt.Printf("i= %d, values = %d\n", i, values[i])
				current.Right = &TreeNode{Val: values[i]}
			}
			i++
			if i >= len(values) {
				return tree
			}
		}
		if current.Right != nil {
			queue = append(queue, current.Right)
		}
	}
	return tree
}
func pathSum(root *TreeNode, targetSum int) [][]int {
	var paths [][]int
	var values []int
	var traverse func(*TreeNode, int)
	traverse = func(node *TreeNode, targetSum int) {
		if node == nil {
			return
		}
		values = append(values, node.Val)
		traverse(node.Left, targetSum-node.Val)
		traverse(node.Right, targetSum-node.Val)
		if node.Left == nil && node.Right == nil && targetSum == node.Val {
			sums2 := make([]int, len(values))
			copy(sums2, values)
			paths = append(paths, sums2)
		}
		values = values[:len(values)-1]
	}
	traverse(root, targetSum)
	return paths
}

func insert_btree(n *TreeNode, v int) *TreeNode {
	if n == nil {
		return &TreeNode{v, nil, nil}
	} else if v <= n.Val {
		n.Left = insert_btree(n.Left, v)
	} else {
		n.Right = insert_btree(n.Right, v)
	}
	return n
}

func flatten(root *TreeNode) {
	var helper func(*TreeNode, *TreeNode) *TreeNode
	helper = func(node *TreeNode, prev *TreeNode) *TreeNode {
		if node == nil {
			return prev
		}
		prev = helper(node.Right, prev)
		prev = helper(node.Left, prev)
		node.Left = nil
		node.Right = prev
		prev = node
		return prev
	}
	helper(root, nil)
}
func dfs_preorder_print(node *TreeNode) {
	if node == nil {
		return
	}
	fmt.Printf(" %d ", node.Val)
	dfs_preorder_print(node.Left)
	dfs_preorder_print(node.Right)
}

// https://leetcode.com/problems/binary-tree-level-order-traversal-ii/submissions/
func levelOrderBottom(root *TreeNode) [][]int {
	if root == nil {
		return nil
	}
	queue := []*TreeNode{root}
	var results [][]int
	for len(queue) > 0 {
		var temp []int
		n := len(queue)
		for i := 0; i < n; i++ {
			node := queue[0]
			queue = queue[1:]
			temp = append(temp, node.Val)
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
		results = append(results, temp)
	}
	for i, j := 0, len(results)-1; i < j; i, j = i+1, j-1 {
		results[i], results[j] = results[j], results[i]
	}
	return results
}

// https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/submissions/
func buildTree(inorder []int, postorder []int) *TreeNode {
	if len(inorder) == 0 {
		return nil
	}
	var mid int
	root := &TreeNode{Val: postorder[len(postorder)-1]}
	for i := range inorder {
		if inorder[i] == root.Val {
			mid = i
			break
		}
	}
	root.Left = buildTree(inorder[:mid], postorder[:mid])
	root.Right = buildTree(inorder[mid+1:], postorder[mid:len(postorder)-1])
	return root
}

// https://leetcode.com/problems/unique-binary-search-trees-ii/

func generateTrees(n int) []*TreeNode {
	var generate func(int, int) []*TreeNode
	generate = func(left int, right int) (result_list []*TreeNode) {
		if left > right {
			return []*TreeNode{nil}
		}
		for v := left; v <= right; v++ {
			left_list := generate(left, v-1)
			right_list := generate(v+1, right)
			for _, left_node := range left_list {
				for _, right_node := range right_list {
					root := &TreeNode{Val: v}
					root.Left = left_node
					root.Right = right_node
					result_list = append(result_list, root)
				}
			}
		}
		return
	}
	return generate(1, n)
}

// https://leetcode.com/problems/unique-binary-search-trees/submissions/
func numTrees(n int) int {
	memo := make(map[int]int)
	var numTree func(int) int
	numTree = func(n int) (num int) {
		if n <= 1 {
			return 1
		}
		if _, ok := memo[n]; ok {
			return memo[n]
		}
		for i := 1; i <= n; i++ {
			num += numTree(i-1) * numTree(n-i)
		}
		memo[n] = num
		return num
	}
	return numTree(n)
}

// https://leetcode.com/problems/interleaving-string/submissions/
func isInterleave(s1 string, s2 string, s3 string) bool {
	memo := make(map[[2]int]bool)
	var helper func(i int, j int, k int) bool
	helper = func(i int, j int, k int) bool {
		if i == len(s1) && j == len(s2) && k == len(s3) {
			return true
		}
		pair := [2]int{i, j}
		if _, ok := memo[pair]; ok {
			return memo[pair]
		}
		result1, result2 := false, false
		if i != len(s1) && k != len(s3) && s1[i] == s3[k] {
			result1 = helper(i+1, j, k+1)
		}
		if j != len(s2) && k != len(s3) && s2[j] == s3[k] {
			result2 = helper(i, j+1, k+1)
		}
		memo[pair] = (result1 || result2)
		return memo[pair]
	}
	return helper(0, 0, 0)
}

// https://leetcode.com/problems/search-insert-position/
func searchInsert(nums []int, target int) int {
	if target > nums[len(nums)-1] {
		return len(nums)
	}
	left, right := 0, len(nums)
	for left <= right {
		mid := int((left + right) / 2)
		if nums[mid] == target {
			return mid
		}
		if nums[mid] > target {
			right = mid - 1
		} else {
			left = mid + 1
		}
	}
	return left
}

// https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/submissions/

func zigzagLevelOrder(root *TreeNode) [][]int {
	if root == nil {
		return nil
	}
	var level []int
	var levels [][]int
	queue := []*TreeNode{root}
	left_direction := true
	for len(queue) > 0 {
		var currentNode *TreeNode
		level = nil
		qlen := len(queue)
		for i := 0; i < qlen; i++ {
			currentNode = queue[0]
			if len(queue) > 1 {
				queue = queue[1:]
			} else {
				queue = nil
			}
			level = append(level, currentNode.Val)
			if currentNode.Left != nil {
				queue = append(queue, currentNode.Left)
			}
			if currentNode.Right != nil {
				queue = append(queue, currentNode.Right)
			}
		}
		if !left_direction {
			for i, j := 0, len(level)-1; i < j; i, j = i+1, j-1 {
				level[i], level[j] = level[j], level[i]
			}
		}
		levels = append(levels, level)
		left_direction = !left_direction
	}
	return levels
}

// https://leetcode.com/problems/populating-next-right-pointers-in-each-node/

type Node struct {
	Val   int
	Left  *Node
	Right *Node
	Next  *Node
}

func connect(root *Node) *Node {
	if root == nil {
		return nil
	}
	queue := []*Node{root}
	for len(queue) > 0 {
		prev := &Node{Val: -999}
		qlen := len(queue)
		for i := 0; i < qlen; i++ {
			currentNode := queue[0]
			if len(queue) > 1 {
				queue = queue[1:]
			} else {
				queue = nil
			}
			if currentNode.Left != nil {
				queue = append(queue, currentNode.Left)
				prev.Next = currentNode.Left
				prev = prev.Next
			}
			if currentNode.Right != nil {
				queue = append(queue, currentNode.Right)
				prev.Next = currentNode.Right
				prev = prev.Next
			}
		}
	}
	return root
}

func connect2(root *Node) *Node {
	if root == nil {
		return nil
	}
	headNode := root

	for headNode != nil {
		dummy := &Node{Val: -999}
		currentNode := dummy
		for headNode != nil {
			if headNode.Left != nil {
				currentNode.Next = headNode.Left
				currentNode = currentNode.Next
			}
			if headNode.Right != nil {
				currentNode.Next = headNode.Right
				currentNode = currentNode.Next
			}
			headNode = headNode.Next
		}
		headNode = dummy.Next
	}
	return root
}

// https://leetcode.com/problems/permutations/solutions/2940507/easy-and-clear-solution-python-3/

func permute(nums []int) [][]int {
	var results [][]int
	var backtrack func([]int, []int)
	backtrack = func(nums []int, path []int) {
		if len(nums) == 0 {
			results = append(results, path)
			return
		}
		for i := 0; i < len(nums); i++ {
			new_path := make([]int, len(path)+1)
			copy(new_path, path)
			new_path[len(new_path)-1] = nums[i]
			new_nums := make([]int, len(nums)-1)
			copy(new_nums, nums[:i])
			copy(new_nums[i:], nums[i+1:])
			backtrack(new_nums, new_path)
		}
	}
	backtrack(nums, []int{})
	return results
}

func permuteUnique(nums []int) [][]int {
	var results [][]int
	var backtrack func([]int, []int)
	backtrack = func(nums []int, path []int) {
		if len(nums) == 0 {
			results = append(results, path)
			return
		}
		visited := make(map[int]bool)
		for i := 0; i < len(nums); i++ {
			if _, ok := visited[nums[i]]; ok {
				continue
			}
			new_path := make([]int, len(path)+1)
			copy(new_path, path)
			new_path[len(new_path)-1] = nums[i]
			new_nums := make([]int, len(nums)-1)
			copy(new_nums, nums[:i])
			copy(new_nums[i:], nums[i+1:])
			backtrack(new_nums, new_path)
			visited[nums[i]] = true
		}
	}
	backtrack(nums, []int{})
	return results
}

// https://leetcode.com/problems/group-anagrams/submissions/
// func groupAnagrams(strs []string) [][]string {
// 	result := make(map[string][]string)
// 	for _, s := range strs {
// 		s_slice := strings.Split(s, "")
// 		sort.Strings(s_slice)
// 		sorted_str := strings.Join(s_slice, "")
// 		if _, ok := result[sorted_str]; !ok {
// 			result[sorted_str] = []string{}
// 		}
// 		result[sorted_str] = append(result[sorted_str], s)
// 	}
// 	var results [][]string
// 	for _, v := range result {
// 		results = append(results, v)
// 	}
// 	return results
// }

// https://leetcode.com/problems/powx-n/submissions/
func myPow(x float64, n int) float64 {
	if n == 0 {
		return 1
	}
	var result float64
	var exp int
	if n < 0 {
		exp = n * (-1)
	} else {
		exp = n
	}
	if n%2 == 0 {
		result = myPow(x*x, int(exp/2))
	} else {
		result = x * myPow(x*x, int((exp-1)/2))
	}
	if n < 0 {
		result = 1 / result
	}
	return result
}

// https://leetcode.com/problems/longest-substring-without-repeating-characters/submissions/
func lengthOfLongestSubstring(s string) int {
	maxLen := 0
	for j := 0; j < len(s); j++ {
		subStr := make(map[rune]bool)
		for _, r := range s[j:] {
			if _, ok := subStr[r]; !ok {
				subStr[r] = true
			} else {
				break
			}
		}
		maxLen = max(maxLen, len(subStr))
	}
	return maxLen
}

// https://leetcode.com/problems/valid-palindrome-ii/submissions/

func validPalindrome(s string) bool {
	// re := regexp.MustCompile(`[^a-zA-Z0-9]`)
	// s = re.ReplaceAllString(strings.ToLower(s),"")
	isPalindrome := func(i int, j int) bool {
		for ; i < j; i, j = i+1, j-1 {
			if s[i] != s[j] {
				return false
			}
		}
		return true
	}
	for i, j := 0, len(s)-1; i < j; i, j = i+1, j-1 {
		if s[i] != s[j] {
			return isPalindrome(i+1, j) || isPalindrome(i, j-1)
		}
	}
	return true

}

// https://leetcode.com/problems/linked-list-cycle-ii/submissions/
func detectCycle(head *ListNode) *ListNode {
	if head == nil {
		return nil
	}
	tortoise, hare := head, head
	for {
		tortoise = tortoise.Next
		hare = hare.Next
		if hare == nil || hare.Next == nil {
			return nil
		} else {
			hare = hare.Next
		}
		if hare == tortoise {
			break
		}
	}
	current1, current2 := head, tortoise
	for current1 != current2 {
		current1 = current1.Next
		current2 = current2.Next
	}
	return current1
}

// https://leetcode.com/problems/kth-largest-element-in-an-array/
func findKthLargest(nums []int, k int) int {
	var quickSort func(int, int) int
	quickSort = func(start int, end int) int {
		// if (end-start)<=1 {
		//     return
		// }
		pivot := end
		i := start
		for j := start; j <= pivot; j++ {
			if nums[j] <= nums[pivot] {
				nums[j], nums[i] = nums[i], nums[j]
				i++
			}
		}
		return i - 1
	}
	var quickSelect func(int, int) int
	quickSelect = func(start int, end int) int {
		i := quickSort(start, end)
		if (len(nums) - k) == i {
			return nums[i]
		}
		if (len(nums) - k) < i {
			return quickSelect(start, i-1)
		} else {
			return quickSelect(i+1, end)
		}
	}
	return quickSelect(0, len(nums)-1)
}

// https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/description/
func searchRange(nums []int, target int) []int {
	binarySearch := func() int {
		left, right := 0, len(nums)-1
		for left <= right {
			mid := int((left + right) / 2)
			if nums[mid] == target {
				return mid
			} else if nums[mid] < target {
				left = mid + 1
			} else {
				right = mid - 1
			}
		}
		return -1
	}
	index := binarySearch()
	if index == -1 {
		return []int{-1, -1}
	}
	left, right := index, index
	for ; left > -1; left-- {
		if nums[left] != nums[index] {
			break
		}
	}
	for ; right < len(nums); right++ {
		if nums[right] != nums[index] {
			break
		}
	}
	// fmt.Printf("nums= %d, target= %d index= %d\n", nums, target, index)
	// fmt.Printf("result= %d\n", []int{left+1, right-1})
	return []int{left + 1, right - 1}
}

// https://leetcode.com/problems/binary-tree-right-side-view/submissions/

func rightSideView(root *TreeNode) []int {
	var traverse func(*TreeNode, int, []int) []int
	traverse = func(node *TreeNode, level int, results []int) []int {
		if node == nil {
			return results
		}
		if len(results) == level {
			results = append(results, node.Val)
		}
		results = traverse(node.Right, level+1, results)
		results = traverse(node.Left, level+1, results)
		return results
	}
	return traverse(root, 0, []int{})
}

// https://leetcode.com/problems/count-complete-tree-nodes/submissions/
func countNodes(root *TreeNode) int {
	completeTreeHeight := func(node *TreeNode) (height int) {
		for height = 0; node.Left != nil; {
			height++
			node = node.Left
		}
		return height
	}
	nodeExists := func(index int, height int, node *TreeNode) bool {
		left, right := 0, int(math.Pow(2, float64(height)))-1
		for count := 0; count < height; {
			mid := int(math.Ceil(float64(left+right) / 2))
			if index >= mid {
				node = node.Right
				left = mid
			} else {
				node = node.Left
				right = mid - 1
			}
			count++
		}
		return node != nil
	}
	if root == nil {
		return 0
	}
	height := completeTreeHeight(root)
	if height == 0 {
		return 1
	}
	upperCount := int(math.Pow(2, float64(height))) - 1
	left, right := 0, upperCount
	for left < right {
		mid := int(math.Ceil(float64(left+right) / 2))
		if nodeExists(mid, height, root) {
			left = mid
		} else {
			right = mid - 1
		}
	}
	// fmt.Printf("result = %d\n", upperCount+left+1)
	return upperCount + left + 1
}

// https://leetcode.com/problems/validate-binary-search-tree/

func isValidBST(root *TreeNode) bool {
	var helper func(*TreeNode, float64, float64) bool
	helper = func(node *TreeNode, low float64, high float64) bool {
		if node == nil {
			return true
		}
		if float64(node.Val) <= low || float64(node.Val) >= high {
			return false
		}
		return helper(node.Left, low, float64(node.Val)) && helper(node.Right, float64(node.Val), high)
	}
	return helper(root, math.Inf(-1), math.Inf(1))
}

type MaxHeapNode struct {
	Vals []int
}

func (heap *MaxHeapNode) Insert(value int) int { // sift up
	heap.Vals = append(heap.Vals, value)
	idx := len(heap.Vals) - 1
	parent_idx := int((idx - 1) / 2)
	for heap.Vals[idx] > heap.Vals[parent_idx] {
		heap.Vals[idx], heap.Vals[parent_idx] = heap.Vals[parent_idx], heap.Vals[idx]
		idx = parent_idx
		parent_idx = int((idx - 1) / 2)
	}
	// fmt.Printf("heap.Vals = %v\n", heap.Vals)
	return idx
}

func (heap *MaxHeapNode) Remove() { // sift down
	if len(heap.Vals) == 0 {
		return
	}
	heap.Vals[0], heap.Vals[len(heap.Vals)-1] = heap.Vals[len(heap.Vals)-1], heap.Vals[0]
	heap.Vals = heap.Vals[:len(heap.Vals)-1]
	idx := 0
	left_idx := idx*2 + 1
	right_idx := idx*2 + 2
	for {
		if heap.Vals[idx] > heap.Vals[left_idx] && heap.Vals[idx] > heap.Vals[right_idx] {
			break
		} else if heap.Vals[idx] < heap.Vals[left_idx] && heap.Vals[left_idx] > heap.Vals[right_idx] {
			heap.Vals[idx], heap.Vals[left_idx] = heap.Vals[left_idx], heap.Vals[idx]
			idx = left_idx
		} else if heap.Vals[idx] < heap.Vals[right_idx] && heap.Vals[right_idx] > heap.Vals[left_idx] {
			heap.Vals[idx], heap.Vals[right_idx] = heap.Vals[right_idx], heap.Vals[idx]
			idx = right_idx
		}
		left_idx = idx*2 + 1
		right_idx = idx*2 + 2
		if right_idx > len(heap.Vals)-1 {
			right_idx = left_idx
		}
	}
	// fmt.Printf("heap.Vals = %v\n", heap.Vals)
}

type MinHeapNode struct {
	Vals []int
}

func (heap *MinHeapNode) Insert(value int) int { // sift up
	heap.Vals = append(heap.Vals, value)
	idx := len(heap.Vals) - 1
	parent_idx := int((idx - 1) / 2)
	for heap.Vals[idx] < heap.Vals[parent_idx] {
		heap.Vals[idx], heap.Vals[parent_idx] = heap.Vals[parent_idx], heap.Vals[idx]
		idx = parent_idx
		parent_idx = int((idx - 1) / 2)
	}
	// fmt.Printf("heap.Vals = %v\n", heap.Vals)
	return idx
}

func (heap *MinHeapNode) Remove() { // sift down
	if len(heap.Vals) == 0 {
		return
	}
	heap.Vals[0], heap.Vals[len(heap.Vals)-1] = heap.Vals[len(heap.Vals)-1], heap.Vals[0]
	heap.Vals = heap.Vals[:len(heap.Vals)-1]
	idx := 0
	left_idx := idx*2 + 1
	right_idx := idx*2 + 2
	for {
		if heap.Vals[idx] < heap.Vals[left_idx] && heap.Vals[idx] < heap.Vals[right_idx] {
			break
		} else if heap.Vals[idx] > heap.Vals[left_idx] && heap.Vals[left_idx] < heap.Vals[right_idx] {
			heap.Vals[idx], heap.Vals[left_idx] = heap.Vals[left_idx], heap.Vals[idx]
			idx = left_idx
		} else if heap.Vals[idx] > heap.Vals[right_idx] && heap.Vals[right_idx] < heap.Vals[left_idx] {
			heap.Vals[idx], heap.Vals[right_idx] = heap.Vals[right_idx], heap.Vals[idx]
			idx = right_idx
		}
		left_idx = idx*2 + 1
		right_idx = idx*2 + 2
		if right_idx > len(heap.Vals)-1 {
			right_idx = left_idx
		}
	}
	// fmt.Printf("heap.Vals = %v\n", heap.Vals)
}

type PriorityQueueItem struct {
	Val      int
	Priority int
}

type PriorityQueue struct {
	Items []*PriorityQueueItem
}

func (pQueue *PriorityQueue) Push(value int, priority int) int { // sift up
	pQueueItem := &PriorityQueueItem{Val: value, Priority: priority}
	pQueue.Items = append(pQueue.Items, pQueueItem)
	idx := len(pQueue.Items) - 1
	parent_idx := int((idx - 1) / 2)
	for pQueue.Items[idx].Priority > pQueue.Items[parent_idx].Priority {
		pQueue.swap(idx, parent_idx)
		idx = parent_idx
		parent_idx = int((idx - 1) / 2)
	}
	// fmt.Printf("pQueue.Vals = %v\n", pQueue.Vals)
	return idx
}

func (pQueue *PriorityQueue) Size() int {
	return len(pQueue.Items)
}

func (pQueue *PriorityQueue) Peek() int {
	return pQueue.Items[0].Val
}

func (pQueue *PriorityQueue) isEmpty() bool {
	return len(pQueue.Items) == 0
}

func (pQueue *PriorityQueue) swap(index1 int, index2 int) {
	pQueue.Items[index1].Priority, pQueue.Items[index2].Priority = pQueue.Items[index2].Priority, pQueue.Items[index1].Priority
}

func (pQueue *PriorityQueue) compare(index1 int, index2 int, index3 int) bool {
	return pQueue.Items[index1].Priority > pQueue.Items[index2].Priority && pQueue.Items[index1].Priority > pQueue.Items[index3].Priority
}

func (pQueue *PriorityQueue) Pop() { // sift down
	if len(pQueue.Items) == 0 {
		return
	}
	pQueue.swap(0, pQueue.Size()-1)
	pQueue.Items = pQueue.Items[:len(pQueue.Items)-1]
	idx := 0
	left_idx := idx*2 + 1
	right_idx := idx*2 + 2
	for {
		if pQueue.compare(idx, left_idx, right_idx) {
			break
		} else if pQueue.compare(left_idx, idx, right_idx) {
			pQueue.swap(idx, left_idx)
			idx = left_idx
		} else if pQueue.compare(right_idx, idx, left_idx) {
			pQueue.swap(idx, right_idx)
			idx = right_idx
		}
		left_idx = idx*2 + 1
		right_idx = idx*2 + 2
		if right_idx > len(pQueue.Items)-1 {
			right_idx = left_idx
		}
	}
	// fmt.Printf("pQueue.Vals = %v\n", pQueue.Vals)
}
