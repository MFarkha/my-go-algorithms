package main

// import "fmt"

// import (
// 	"fmt"
// 	"sort"
// )

func main() {

	// myLinkedList1 := createLinkedList(1)
	// myLinkedList1.prepend(0)
	// myLinkedList1.append(2)
	// myLinkedList1.append(40)
	// myLinkedList1.append(50)
	// fmt.Print(myLinkedList1)

	// myLinkedList1.insert(3, 3)
	// fmt.Print(myLinkedList1)

	// myLinkedList1.remove(4)
	// fmt.Print(myLinkedList1)

	// myLinkedList1.reverse()
	// fmt.Print(myLinkedList1)

	// myHashTable1 := createHashTable(5)
	// myHashTable1.set("apples", "15")
	// myHashTable1.set("grapes", "25")
	// myHashTable1.set("oranges", "35")

	// fmt.Println(myHashTable1.get("oranges"))

	// for _, k := range myHashTable1.keys() {
	// 	fmt.Printf("%s ", k)
	// }
	// fmt.Println()

	// basket := []int{2, 65, 34, 2, 1, 7, 8}
	// // sort.Ints(basket)
	// sort.Strings(basket)
	// fmt.Println(basket)
	// 2,4,3
	// 5,6,4

	// linked_list1 := []int{2, 4, 3, 0, 0, 0, 1}
	// linked_list2 := []int{5, 7, 4}

	// // linked_list1 := []int{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}
	// // linked_list2 := []int{5, 6, 4}

	// rootNode1 := &ListNode{Val: linked_list1[0]}
	// rootNode2 := &ListNode{Val: linked_list2[0]}
	// currentNode := rootNode1
	// for _, v := range linked_list1[1:] {
	// 	currentNode.Next = &ListNode{Val: v}
	// 	currentNode = currentNode.Next
	// }

	// currentNode = rootNode2
	// for _, v := range linked_list2[1:] {
	// 	currentNode.Next = &ListNode{Val: v}
	// 	currentNode = currentNode.Next
	// }

	// addTwoNumbers(rootNode1, rootNode2)

	// s: = "III"
	// s: = "MCMXCIV"
	// romanToInt(s)

	// s := "()"
	// fmt.Println(isValid(s))
	// a := "100"
	// b := "110010"
	// fmt.Println(addBinary(a, b))

	// tree1 := &TreeNode{
	// 	Val:   5,
	// 	Left:  &TreeNode{Val: 4, Left: &TreeNode{Val: 11, Left: &TreeNode{Val: 7, Left: nil, Right: nil}, Right: &TreeNode{Val: 2, Left: nil, Right: nil}}, Right: nil},
	// 	Right: &TreeNode{Val: 8, Left: &TreeNode{Val: 13, Left: nil, Right: nil}, Right: &TreeNode{Val: 4, Left: &TreeNode{Val: 1, Left: nil, Right: nil}, Right: &TreeNode{Val: 5, Left: nil, Right: nil}}}}
	// targetSum := 22
	// fmt.Printf("paths= %v\n", pathSum(tree1, targetSum))
	// tree1 := &TreeNode{Val: 0}
	// flatten(tree1)
	// fmt.Println()
	// dfs_preorder_print(tree1)
	// fmt.Println()

	// nums := []int{1, 2, 3}
	// fmt.Printf("results = %v\n", permute(nums))

	// nums := []int{3, 2, 1, 5, 6, 4}
	// k := 2
	// fmt.Println(findKthLargest(nums, k))

	// nums := []int{5, 7, 7, 8, 8, 10}
	// target := 7
	// searchRange(nums, target)

	// tree2 := &TreeNode{Val: 1}
	// tree2.insert([]int{2, 3, 4, 5, 6, 0})
	// countNodes(tree2)

	// tree3 := &TreeNode{Val: 2}
	// tree3.insert([]int{1, 3})
	// isValidBST(tree3)

	heap := &MaxHeapNode{Vals: []int{75, 50, 25, 45, 35, 10, 15, 20, 40}}
	// heap.Insert(45)
	heap.Remove()

}
