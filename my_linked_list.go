package main

import (
	"fmt"
	"log"
)

type Node struct {
	value int
	next  *Node
}

type Linked_List struct {
	head   *Node
	tail   *Node
	length int
}

func createNode(value int) *Node {
	var newNode Node
	newNode.value = value
	newNode.next = nil
	return &newNode
}

func createLinkedList(value int) (ll Linked_List) {
	ll.head = createNode(value)
	ll.tail = ll.head
	ll.length = 1
	return ll
}

func (ll Linked_List) String() string {
	output := fmt.Sprintf("Linked List (length:%d)\n", ll.length)
	currentNode := ll.head
	for currentNode != nil {
		output += fmt.Sprintf("\tcurrentNode.value = %d\n", currentNode.value)
		currentNode = currentNode.next
	}
	return output
}

func (ll *Linked_List) prepend(value int) {
	newNode := createNode(value)
	newNode.next = ll.head
	ll.head = newNode
	ll.length++
}

func (ll *Linked_List) append(value int) {
	newNode := createNode(value)
	ll.tail.next = newNode
	ll.tail = newNode
	ll.length++
}

func (ll *Linked_List) traverseToIndex(index int) *Node {
	currentNode := ll.head
	for i := 0; i < index; i++ {
		currentNode = currentNode.next
	}
	return currentNode
}

func (ll *Linked_List) insert(index int, value int) {
	if index < 0 {
		return
	}
	newNode := createNode(value)
	if index == 0 {
		newNode.next = ll.head
		ll.head = newNode
		ll.length++
		return
	}
	if index >= ll.length {
		ll.tail.next = newNode
		ll.tail = newNode
		ll.length++
		return
	}
	currentNode := ll.traverseToIndex(index - 1)
	nextNode := currentNode.next
	currentNode.next = newNode // append for currentNode
	newNode.next = nextNode    // prepend for nextNode
	ll.length++
}

func (ll *Linked_List) remove(index int) {
	if ll.length == 1 {
		log.Fatal("The Linked List is one element only. Nothing to delete")
	}
	if index >= ll.length || index < 0 {
		log.Fatal("The index is more than length or less than zero")
	}
	if index == 0 {
		ll.head = ll.head.next // removing current head
		if ll.head == nil || ll.head.next == nil {
			ll.tail = ll.head
		}
	} else {
		currentNode := ll.traverseToIndex(index - 1)
		unwantedNode := currentNode.next
		currentNode.next = unwantedNode.next // removing unwantedNode
		if currentNode.next == nil {
			ll.tail = currentNode
		}
	}
	ll.length -= 1
}

func (ll *Linked_List) reverse() {
	if ll.length <= 1 {
		log.Fatal("The Linked List is one element only. Nothing to reverse")
		return //nothing to do - less or one item only list
	}
	firstNode := ll.head
	secondNode := firstNode.next
	ll.tail = ll.head
	for secondNode != nil {
		tempNode := secondNode.next
		secondNode.next = firstNode
		firstNode = secondNode
		secondNode = tempNode
	}
	ll.head.next = nil
	ll.head = firstNode
}
